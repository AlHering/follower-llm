# -*- coding: utf-8 -*-
"""
****************************************************
*          Basic Language Model Backend            *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import uvicorn
from enum import Enum
import traceback
from typing import Optional, Any
from datetime import datetime as dt
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from functools import wraps
from src.configuration import configuration as cfg
from src.control.follower_llm_controller import FollowerLLMController, FilterMask

"""
Backend control
"""
BACKEND = FastAPI(title="LLM Follower Backend", version="0.1",
                  description="Backend for serving LLM Follower services.")
CONTROLLER: FollowerLLMController = FollowerLLMController()


def interface_function() -> Optional[Any]:
    """
    Validation decorator.
    :param func: Decorated function.
    :return: Error message if status is incorrect, else function return.
    """
    global CONTROLLER

    def wrapper(func: Any) -> Optional[Any]:
        """
        Function wrapper.
        :param func: Wrapped function.
        :return: Error message if status is incorrect, else function return.
        """
        @wraps(func)
        async def inner(*args: Optional[Any], **kwargs: Optional[Any]):
            """
            Inner function wrapper.
            :param args: Arguments.
            :param kwargs: Keyword arguments.
            """
            requested = dt.now()
            try:
                response = await func(*args, **kwargs)
                response["status"] = "successful"
            except Exception as ex:
                response = {
                    "status": "failed",
                    "exception": ex,
                    "trace": traceback.format_exc()
                }
            responded = dt.now()
            CONTROLLER.post_object(
                "log",
                request={
                    "function": func.__name__,
                    "args": args,
                    "kwargs": kwargs
                },
                response=response,
                requested=requested,
                responded=responded
            )
            return response
        return inner
    return wrapper


"""
Dataclasses
"""

# TODO: Allow for None values in respective fields


class LanguageModelInstance(BaseModel):
    """
    Dataclass for language model instance objects.
    """
    backend: str
    model_path: str

    model_file: str
    model_parameters: dict
    tokenizer_path: str
    tokenizer_parameters: dict
    config_path: str
    config_parameters: dict

    default_system_prompt: str
    use_history: bool
    encoding_parameters: dict
    generating_parameters: dict
    decoding_parameters: dict

    resource_requirements: dict


class KnowledgebaseInstance(BaseModel):
    """
    Dataclass for knowledgebase objects.
    """
    backend: str
    knowledgebase_path: str
    knowledgebase_parameters: dict
    preprocessing_parameters: dict
    embedding_parameters: dict
    retrieval_parameters: dict


"""
Backend endpoint enumerator
"""


class Endpoints(str, Enum):
    """
    String-based endpoint enum class.
    """
    BASE = "/api/v1"

    GET_LMS = f"{BASE}/lms/"
    GET_LM = f"{BASE}/lm/get/{{lm_instance_id}}"
    POST_LM = f"{BASE}/lm/post"
    PUT_LM = f"{BASE}/lm/put"
    PATCH_LM = f"{BASE}/lm/patch/{{lm_instance_id}}"
    DELETE_LM = f"{BASE}/lm/delete/{{lm_instance_id}}"

    GET_KBS = f"{BASE}/kbs/"
    GET_KB = f"{BASE}/kb/get/{{kb_instance_id}}"
    POST_KB = f"{BASE}/kb/post"
    PUT_KB = f"{BASE}/kb/put"
    PATCH_KB = f"{BASE}/kb/patch/{{kb_instance_id}}"
    DELETE_KB = f"{BASE}/kb/delete/{{kb_instance_id}}"

    EMBED_DOCUMENT = f"{BASE}/kbs/{{kb_instance_id}}/embed/{{file_id}}"
    DELETE_DOCUMENT = f"{BASE}/kbs/{{kb_instance_id}}/delete/{{file_id}}"

    POST_QUERY = f"{BASE}/query"

    def __str__(self) -> str:
        """
        Getter method for a string representation.
        """
        return self.value


"""
Endpoints
"""


"""
Language model instances
"""


@BACKEND.get(Endpoints.GET_LMS)
@interface_function()
async def get_lm_instances() -> dict:
    """
    Endpoint for getting language model instances.
    :return: Response.
    """
    global CONTROLLER
    return {"lms": CONTROLLER.get_objects_by_type("lminstance")}


@BACKEND.get(Endpoints.GET_LM)
@interface_function()
async def get_lm_instance(lm_instance_id: int) -> dict:
    """
    Endpoint for getting a language model instance.
    :param lm_instance_id: LM instance ID.
    :return: Response.
    """
    global CONTROLLER
    return {"lm": CONTROLLER.get_object_by_id("lminstance", lm_instance_id)}


@BACKEND.post(Endpoints.POST_LM)
@interface_function()
async def post_lm_instance(lm_instance: LanguageModelInstance) -> dict:
    """
    Endpoint for posting a language model instance.
    :param lm_instance: LM instance.
    :return: Response.
    """
    global CONTROLLER
    return {"lm": CONTROLLER.post_object("lminstance", **dict(lm_instance))}


@BACKEND.put(Endpoints.PUT_LM)
@interface_function()
async def put_lm_instance(lm_instance: LanguageModelInstance) -> dict:
    """
    Endpoint for putting in a language model instance.
    :param lm_instance: LM instance.
    :return: Response.
    """
    global CONTROLLER
    return {"lm": CONTROLLER.put_object("lminstance", **dict(lm_instance))}


@BACKEND.patch(Endpoints.PATCH_LM)
@interface_function()
async def patch_lm_instance(lm_instance_id: int, lm_instance: LanguageModelInstance) -> dict:
    """
    Endpoint for patching a language model instance.
    :param lm_instance_id: LM instance ID.
    :param lm_instance: LM instance.
    :return: Response.
    """
    global CONTROLLER
    return {"lm": CONTROLLER.patch_object("lminstance", lm_instance_id, **dict(lm_instance))}


"""
Knowledgebases
"""


@BACKEND.get(Endpoints.GET_KBS)
@interface_function()
async def get_kb_instances() -> dict:
    """
    Endpoint for getting knowledgebase instances.
    :return: Response.
    """
    global CONTROLLER
    return {"kbs": CONTROLLER.get_objects_by_type("kbinstance")}


@BACKEND.get(Endpoints.GET_KB)
@interface_function()
async def get_kb_instance(kb_instance_id: int) -> dict:
    """
    Endpoint for getting a knowledgebase instance.
    :param kb_instance_id: KB instance ID.
    :return: Response.
    """
    global CONTROLLER
    return {"kb": CONTROLLER.get_object_by_id("kbinstance", kb_instance_id)}


@BACKEND.post(Endpoints.POST_KB)
@interface_function()
async def post_kb_instance(kb_instance: KnowledgebaseInstance) -> dict:
    """
    Endpoint for posting a knowledgebase instance.
    :param kb_instance: KB instance.
    :return: Response.
    """
    global CONTROLLER
    return {"kb": CONTROLLER.post_object("kbinstance", **dict(kb_instance))}


@BACKEND.put(Endpoints.PUT_KB)
@interface_function()
async def put_kb_instance(kb_instance: KnowledgebaseInstance) -> dict:
    """
    Endpoint for putting in a knowledgebase instance.
    :param kb_instance: KB instance.
    :return: Response.
    """
    global CONTROLLER
    return {"kb": CONTROLLER.put_object("kbinstance", **dict(kb_instance))}


@BACKEND.patch(Endpoints.PATCH_KB)
@interface_function()
async def patch_kb_instance(kb_instance_id: int, kb_instance: KnowledgebaseInstance) -> dict:
    """
    Endpoint for patching a knowledgebase instance.
    :param kb_instance_id: KB instance ID.
    :param kb_instance: KB instance.
    :return: Response.
    """
    global CONTROLLER
    return {"kb": CONTROLLER.patch_object("kbinstance", kb_instance_id, **dict(kb_instance))}


"""
Interaction
"""


@BACKEND.post(Endpoints.EMBED_DOCUMENT)
@interface_function()
async def embed_document(kb_instance_id: int, file_id: int,) -> dict:
    """
    Endpoint for embedding a document.
    :param kb_instance_id: KB instance ID.
    :param file_id: File ID.
    :return: Response.
    """
    raise NotImplementedError(
        "Endpoint for embedding document is not yet implemented.")


@BACKEND.delete(Endpoints.DELETE_DOCUMENT)
@interface_function()
async def delete_document(kb_instance_id: int, file_id: int,) -> dict:
    """
    Endpoint for deleting a document embedding.
    :param kb_instance_id: KB instance ID.
    :param file_id: File ID.
    :return: Response.
    """
    raise NotImplementedError(
        "Endpoint for deleting document embeddings is not yet implemented.")


@BACKEND.post(Endpoints.POST_QUERY)
@interface_function()
async def post_query(lm_instance_id: int, kb_instance_id: int, query: str, include_sources: bool = True) -> dict:
    """
    Endpoint for posting document qa query.
    :param lm_instance_id: LM instance ID.
    :param kb_instance_id: KB instance ID.
    :param query: Query.
    :param include_sources: Flag declaring, whether to include sources.
    :return: Response.
    """
    raise NotImplementedError(
        "Endpoint for posting a query is not yet implemented.")


"""
Backend runner
"""


def run_backend(host: str = None, port: int = None, reload: bool = True) -> None:
    """
    Function for running backend server.
    :param host: Server host. Defaults to None in which case "127.0.0.1" is set.
    :param port: Server port. Defaults to None in which case either environment variable "BACKEND_PORT" is set or 7861.
    :param reload: Reload flag for server. Defaults to True.
    """
    uvicorn.run("src.interfaces.backend_interface:BACKEND",
                host="127.0.0.1" if host is None else host,
                port=int(
                    cfg.ENV.get("BACKEND_PORT", 7861) if port is None else port),
                reload=False)


if __name__ == "__main__":
    run_backend()
