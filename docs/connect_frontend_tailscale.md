# Connecting Frontend to Tailscale

## Command Overview

Use the following command to start your frontend application and bind it to a specific Tailscale IP address:

```bash
uvicorn front_end.test:app --host 100.116.152.80 --port 8000
```

### Explanation of the Command

- `uvicorn`: The ASGI server used to serve your Python application.
- `front_end.test:app`: Specifies the Python module (`front_end.test`) and the ASGI application instance (`app`).
- `--host 100.116.152.80`: Binds the server to the specified Tailscale IP address.
- `--port 8000`: Specifies the port on which the server will listen.