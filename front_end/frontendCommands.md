# Frontend Startup Guide

This guide explains how to start the services required for running the frontend applications.

## Starting the Frontend

1. **Start the Neo4j database**  
   This command starts the Neo4j service:

   ```bash
   sudo systemctl start neo4j
   ```

2. ## Run the Streamlit application

    Navigate to your project directory and run:

    ```bash
    streamlit run front_end/streamlit/app.py
    ```

3. ## Launch the Voila dashboard

    Use the following command to run the Voila notebook with custom settings:

    ```bash
    voila front_end/voila/yfiles_graph.ipynb --Voila.ip=0.0.0.0 --Voila.tornado_settings="{'headers':{'Content-Security-Policy':'frame-ancestors *'}}"
    ```

## Notes

* Make sure all paths are correct and match your project structure.
* If any service fails to start, ensure dependencies are installed and permissions are set properly.
