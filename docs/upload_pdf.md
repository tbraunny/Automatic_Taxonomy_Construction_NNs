# How to: upload a PDF to RAG Server

1. Obtain the local <file-path.pdf> to the PDF to upload
2. Obtain the <desktop-ip> to the server from tailscale
3. Use the following command to upload the PDF to the server
    curl -X POST http://<desktop-ip>:5000/api/upload \
     -F "pdf=@<file-path>"
4. The file will now be located in 'data/raw'
