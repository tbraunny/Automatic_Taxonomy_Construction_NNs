# Setup Instructions


1. Download Tailscale and Set Up Tailscale VPN
 - Download Tailscale VPN
 - Join the VPN with the link: https://login.tailscale.com/admin/invite/79sRmCzExZb
 - Obtain the wpeb-436-21 ip connected

2. Refer to the query_rag.py utility script and it's example usage



# Server Setup

## Windows to WSL Port Forwarding

1. Obtain <wsl-ip>:
    ip addr show eth0
    It will look something like inet 172.20.240.1

2. Add the following port forwarding rules
    netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=11434 connectaddress=<wsl-ip> connectport=11434
    netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=22 connectaddress=172.24.220.223 connectport=22

3. Test Connection (optional)
    Test hosting on a port using:
        python3 -m http.server 11434 --bind 0.0.0.0
    Test the connection:
        curl http://<tailscale-ip>:11434



## Ollama Setup and Configuration

1. Set the environment variable for Ollama:
    export OLLAMA_HOST=http://0.0.0.0:11434

2. Start Ollama:
    ollama serve



## Note

1. WSL IP may change per reboot

2. Ollama in **linux** by default installs as a systemd service. 

    Meaning, ollama is always running in the background 
        stop the service with "sudo systemctl stop ollama.service" 
        and disable the service with "sudo systemctl disable ollama.service"
        Now, ollama serve will work