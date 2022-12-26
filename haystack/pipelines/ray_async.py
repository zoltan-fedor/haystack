from __future__ import annotations
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from copy import deepcopy
from pathlib import Path

try:
    from ray import serve
    import ray
except:
    ray = None  # type: ignore
    serve = None  # type: ignore

from haystack.pipelines.config import (
    get_component_definitions,
    get_pipeline_definition,
    read_pipeline_config_from_yaml,
    validate_config,
)
from haystack.nodes.base import BaseComponent, RootNode
from haystack.pipelines.ray import RayPipeline, _RayDeploymentWrapper


class AsyncRayPipeline(RayPipeline):
    @classmethod
    def _create_ray_deployment(
        cls, component_name: str, pipeline_config: dict, serve_deployment_kwargs: Optional[Dict[str, Any]] = {}
    ):
        """
        Create a Ray Deployment for the Component.

        :param component_name: Class name of the Haystack Component.
        :param pipeline_config: The Pipeline config YAML parsed as a dict.
        :param serve_deployment_kwargs: An optional dictionary of arguments to be supplied to the
                                        `ray.serve.deployment()` method, like `num_replicas`, `ray_actor_options`,
                                        `max_concurrent_queries`, etc. See potential values in the
                                         Ray Serve API docs (https://docs.ray.io/en/latest/serve/package-ref.html)
                                         under the `ray.serve.deployment()` method
        """
        RayDeployment = serve.deployment(
            _AsyncRayDeploymentWrapper, name=component_name, **serve_deployment_kwargs  # type: ignore
        )
        RayDeployment.deploy(pipeline_config, component_name)
        handle = RayDeployment.get_handle()
        return handle

    async def _run_node(self, node_id: str, node_input: Dict[str, Any]) -> Tuple[Dict, str]:  # type: ignore
        return await self.graph.nodes[node_id]["component"].remote(**node_input)


class _AsyncRayDeploymentWrapper(_RayDeploymentWrapper):
    """
    Ray Serve supports calling of __init__ methods on the Classes to create "deployment" instances.

    In case of Haystack, some Components like Retrievers have complex init methods that needs objects
    like Document Stores.

    This wrapper class encapsulates the initialization of Components. Given a Component Class
    name, it creates an instance using the YAML Pipeline config.
    """

    node: BaseComponent

    def __init__(self, pipeline_config: dict, component_name: str):
        """
        Create an instance of Component.

        :param pipeline_config: Pipeline YAML parsed as a dict.
        :param component_name: Component Class name.
        """
        self.name = component_name
        if component_name in ["Query", "File"]:
            self.node = RootNode()
        else:
            self.node = self.load_from_pipeline_config(pipeline_config, component_name)

    async def __call__(self, *args, **kwargs):
        """
        Ray calls this method which is then re-directed to the corresponding component's run().
        """
        return await self.node._dispatch_run_general(*args, **kwargs)

    # this is an async version fo the `nodes/base.py::_dispatch_run_general` method
    async def _dispatch_run_general(self, run_method: Callable, **kwargs):
        """
        This method takes care of the following:
          - inspect run_method's signature to validate if all necessary arguments are available
          - pop `debug` and sets them on the instance to control debug output
          - call run_method with the corresponding arguments and gather output
          - collate `_debug` information if present
          - merge component output with the preceding output and pass it on to the subsequent Component in the Pipeline
        """
        arguments = deepcopy(kwargs)
        params = arguments.get("params") or {}

        run_signature_args = inspect.signature(run_method).parameters.keys()

        run_params: Dict[str, Any] = {}
        for key, value in params.items():
            if key == self.name:  # targeted params for this node
                if isinstance(value, dict):
                    # Extract debug attributes
                    if "debug" in value.keys():
                        self.debug = value.pop("debug")

                    for _k, _v in value.items():
                        if _k not in run_signature_args:
                            raise Exception(f"Invalid parameter '{_k}' for the node '{self.name}'.")

                run_params.update(**value)
            elif key in run_signature_args:  # global params
                run_params[key] = value

        run_inputs = {}
        for key, value in arguments.items():
            if key in run_signature_args:
                run_inputs[key] = value

        output, stream = await run_method(**run_inputs, **run_params)

        # Collect debug information
        debug_info = {}
        if getattr(self, "debug", None):
            # Include input
            debug_info["input"] = {**run_inputs, **run_params}
            debug_info["input"]["debug"] = self.debug
            # Include output, exclude _debug to avoid recursion
            filtered_output = {key: value for key, value in output.items() if key != "_debug"}
            debug_info["output"] = filtered_output
        # Include custom debug info
        custom_debug = output.get("_debug", {})
        if custom_debug:
            debug_info["runtime"] = custom_debug

        # append _debug information from nodes
        all_debug = arguments.get("_debug", {})
        if debug_info:
            all_debug[self.name] = debug_info
        if all_debug:
            output["_debug"] = all_debug

        # add "extra" args that were not used by the node, but not the 'inputs' value
        for k, v in arguments.items():
            if k not in output.keys() and k != "inputs":
                output[k] = v

        output["params"] = params
        return output, stream
