"""Robot-local MCP tools, snapshot JSON and continuous NDJSON."""
from contextlib import asynccontextmanager, aclosing
import argparse
import json
import logging
from urllib.parse import urlencode
from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent, ImageContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Mount
import uvicorn
from core import CameraBroker, SharedSource, channels_mask, depth_mask, DEFAULT_CHANNELS

class RequestBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    channels: list[StrictInt] = Field(default_factory=lambda: DEFAULT_CHANNELS.copy(), max_length=64)
    depth_channels: list[StrictInt] | None = Field(default=None,max_length=64)
    instruction: str = Field(default='', max_length=4096)

def make_app(broker, port=8796):
    mcp = FastMCP('Vision Tool', host='127.0.0.1', port=port, stateless_http=True,
                  json_response=True, max_request_body_size=16384,
                  instructions='Read-only155 cameras. 0=head, 1=right wrist. Default all off. '
                  'Streaming does not invoke a model or move the robot.')
    readonly = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
    @mcp.tool(annotations=readonly)
    async def camera_snapshot(instruction: str = '', channels: list[StrictInt] | None = None, depth_channels: list[StrictInt] | None = None) -> CallToolResult:
        """Get fresh selected images: 0=head, 1=right wrist, trailing bits zero, default all off.
        Images include capture/send timestamps. Instruction is context, not a motor command.
        """
        body = RequestBody(channels=channels_mask(channels),depth_channels=depth_channels,instruction=instruction)
        depth_mask(body.depth_channels,body.channels)
        if len(broker.subscriptions) >= 8:
            raise ValueError('Camera consumer capacity reached')
        result = await broker.snapshot(body.channels, body.instruction,depth_channels=body.depth_channels)
        metadata = {**result, 'frames': [{k: v for k, v in f.items() if k != 'image_base64'}
                                       for f in result['frames']]}
        content = [TextContent(type='text', text=json.dumps(metadata, ensure_ascii=False))]
        for frame in result['frames']:
            content.append(ImageContent(type='image', data=frame['image_base64'], mimeType='image/jpeg'))
        return CallToolResult(content=content, structuredContent=metadata, isError=bool(result['errors'] or result.get('depth_errors')))
    @mcp.tool(annotations=readonly)
    async def camera_stream(instruction: str = '', channels: list[StrictInt] | None = None, depth_channels: list[StrictInt] | None = None) -> CallToolResult:
        """Get local continuous NDJSON image feed URL; consumer opens it to begin output.
        10fps/channel until HTTP disconnect. Change channels by reopening the URL.
        This finite MCP call does not inject every video frame into LLM context.
        """
        body = RequestBody(channels=channels_mask(channels),depth_channels=depth_channels,instruction=instruction)
        depth_mask(body.depth_channels,body.channels)
        query = urlencode(dict(channels=json.dumps(body.channels), instruction=body.instruction,depth_channels=json.dumps(depth_mask(body.depth_channels,body.channels))))
        result = dict(url=f'http://127.0.0.1:{port}/v1/stream?{query}', method='GET',
                    content_type='application/x-ndjson', channels=body.channels,
                    fps_per_channel=broker.fps, instruction=body.instruction,
                    activation='on_http_reader_connect', stop='close_http_stream',
                    runtime_host='155', all_zero_policy='status_only_no_camera_poll')
        return CallToolResult(content=[TextContent(type='text', text=json.dumps(result))],
                              structuredContent=result)
    async def parse(request):
        origin = request.headers.get('origin')
        if origin and origin not in (f'http://127.0.0.1:{port}', f'http://localhost:{port}'):
            raise ValueError('Cross-origin camera access is not permitted')
        if request.method == 'GET':
            values = dict(channels=json.loads(request.query_params.get('channels', json.dumps(DEFAULT_CHANNELS))),
                          instruction=request.query_params.get('instruction', ''),
                          depth_channels=json.loads(request.query_params.get('depth_channels','null')))
        else:
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > 16384:
                    raise ValueError('Request exceeds16KiB')
            values = json.loads(raw or b'{}')
        body = RequestBody.model_validate(values)
        channels_mask(body.channels)
        depth_mask(body.depth_channels,body.channels)
        return body
    @mcp.custom_route('/health', methods=['GET'])
    async def health(request):
        return JSONResponse(broker.status())
    @mcp.custom_route('/v1/snapshot', methods=['POST'])
    async def snapshot(request):
        try:
            body = await parse(request)
        except (ValueError, TypeError) as error:
            return JSONResponse({'error': str(error)}, status_code=400)
        if len(broker.subscriptions) >= 8:
            return JSONResponse({'error': 'Camera consumer capacity reached'}, status_code=429)
        return JSONResponse(await broker.snapshot(body.channels, body.instruction,depth_channels=body.depth_channels),
                            headers={'Cache-Control': 'no-store'})
    @mcp.custom_route('/v1/stream', methods=['GET', 'POST'])
    async def stream(request):
        try:
            body = await parse(request)
        except (ValueError, TypeError) as error:
            return JSONResponse({'error': str(error)}, status_code=400)
        if len(broker.subscriptions) >= 8:
            return JSONResponse({'error': 'Camera consumer capacity reached'}, status_code=429)
        async def output():
            async with aclosing(broker.stream(body.channels, body.instruction,depth_channels=body.depth_channels)) as events:
                async for event in events:
                    yield json.dumps(event, ensure_ascii=False, separators=(',', ':')).encode()+b'\n'
        return StreamingResponse(output(), media_type='application/x-ndjson',
                                 headers={'Cache-Control': 'no-store', 'X-Accel-Buffering': 'no'})
    mcp_app = mcp.streamable_http_app()
    @asynccontextmanager
    async def lifespan(app):
        await broker.start()
        try:
            async with mcp.session_manager.run():
                yield
        finally:
            await broker.close()
    return Starlette(routes=[Mount('/', app=mcp_app)], lifespan=lifespan)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8796)
    parser.add_argument('--source-port', type=int, default=8792)
    args = parser.parse_args()
    broker = CameraBroker(SharedSource(f'http://127.0.0.1:{args.source_port}'), fps=10.)
    app = make_app(broker, args.port)
    # FastMCP enables INFO logging; a shared camera GET on every frame otherwise
    # fills the system journal during an unattended display session.
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('httpcore').setLevel(logging.WARNING)
    uvicorn.run(app, host='127.0.0.1', port=args.port,
                log_level='warning', access_log=False, timeout_graceful_shutdown=4)
if __name__ == '__main__':
    main()
