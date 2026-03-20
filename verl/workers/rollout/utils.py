# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import contextlib
import logging
import os

import uvicorn
from fastapi import FastAPI

from verl.utils.net_utils import get_free_port

logger = logging.getLogger(__file__)


def get_max_position_embeddings(hf_config) -> int:
    max_len = getattr(hf_config, "max_position_embeddings", None)
    if max_len is None:
        text_config = getattr(hf_config, "text_config", None)
        if text_config is not None:
            max_len = getattr(text_config, "max_position_embeddings", None)

    if max_len is None:
        raise ValueError("max_position_embeddings not found in HFModelConfig!")
    return int(max_len)


async def run_unvicorn(
    app: FastAPI,
    server_args,
    server_address,
    max_retries: int = 5,
    startup_timeout_s: float = 30.0,
) -> tuple[int, asyncio.Task]:
    server_port, server_task = None, None

    for i in range(max_retries):
        server = None
        try:
            # Reserve a candidate port, then release the socket before uvicorn binds.
            server_port, sock = get_free_port(server_address)
            sock.close()
            app.server_args = server_args
            config = uvicorn.Config(app, host=server_address, port=server_port, log_level="warning")
            server = uvicorn.Server(config)
            server_task = asyncio.create_task(server.serve())

            # Wait until uvicorn actually starts listening.
            deadline = asyncio.get_running_loop().time() + startup_timeout_s
            while not server.started:
                if server_task.done():
                    exc = server_task.exception()
                    if exc is not None:
                        raise exc
                    raise RuntimeError("Uvicorn server exited before reporting ready state")
                if asyncio.get_running_loop().time() >= deadline:
                    raise TimeoutError(f"Timed out waiting for HTTP server startup on port {server_port}")
                await asyncio.sleep(0.05)
            break
        except (OSError, SystemExit, TimeoutError, RuntimeError) as e:
            logger.error(f"Failed to start HTTP server on port {server_port} at try {i}, error: {e}")
            if server is not None:
                server.should_exit = True
            if server_task is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(server_task, timeout=2)
                server_task = None
            server_port = None
            await asyncio.sleep(0.2)
    else:
        logger.error(f"Failed to start HTTP server after {max_retries} retries, exiting...")
        os._exit(-1)

    logger.info(f"HTTP server started on port {server_port}")
    return server_port, server_task


async def ensure_async_iterator(iterable):
    """Convert an iterable to an async iterator."""
    if hasattr(iterable, "__aiter__"):
        async for item in iterable:
            yield item
    else:
        for item in iterable:
            yield item
