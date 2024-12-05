const app = Vue.createApp({
    data () {
        return {
            showAbout: false,
            showTaxonomy: false,
            showOntology: false,
            definitionTitle: "Layer Definitions",
            layerInfo: "Click on a layer to see its definition"
        };
    },

    methods: {
        toggleAbout() {
            this.showAbout = !this.showAbout;
        },
        generateOntology() {
            this.showOntology = true;
            this.showTaxonomy = false;
        },
        generateTaxonomy() {
            this.showTaxonomy = true;
            this.showOntology = false;
        },
        showDefinition(type) {
            this.definitionTitle = `${type} Layer`;
            this.layerInfo = `This is the definition of the ${type} layer.`;
        },
        uploadFile(event) {
            const fileInput = event.target;
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);

            fetch('/api', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                console.log('File uploaded successfully:', data);
            })
            .catch(error => {
                console.error('Error uploading file:', error);
            });
        }
    }

});

app.mount('#app')


async function fetchRoot() {
    const response = await fetch('http://127.0.0.1:8000/');
    const data = await response.json();
    document.getElementById('hello-message').innerText = `Hello message: ${data.Hello}`;
}

//Define the function to display layer definitions
function layer_type(layer) {
    const layerInfo = document.getElementById("layerInfo");
    const definitionTitle = document.getElementById("definition-title");
    const responses = document.getElementById("responses")

    //Definitions for each layer type
    const layerDefinitions = {
        "Convolutional": "A Convolutional Layer applies a convolution operation to the input, passing the result to the next layer. It’s used primarily for feature extraction in image data.",
        "Pooling": "A Pooling Layer reduces the spatial size of the representation to decrease the number of parameters and computation in the network, controlling overfitting.",
        "Fully Connected": "A Fully Connected Layer connects every neuron in one layer to every neuron in another layer, allowing complex patterns and associations to be learned.",
        "ReLU": "A ReLU (Rectified Linear Unit) Layer applies an activation function, introducing non-linearity to help the model learn complex patterns.",
        "Hinge": "Penalty increases the more incorrect the prediction is.",
        "Logistic": "All predictions labeled as mistakes, however some are less of a mistake than others.",
        "Exponential": "Grows faster when mistakes are made, helps ensure no catastrophically incorrect predictions.",
    };

    //responses.style.display = 'block'

    if (responses.style.display === 'block') {
        responses.style.display = 'none';  // Hide the image if it's currently visible
    } else {
        responses.style.display = 'block'; // Show the image if it's currently hidden
    }

    //Update the title and display the corresponding definition
    definitionTitle.innerText = `${layer} Definition`;
    layerInfo.innerText = layerDefinitions[layer];
}

function fetchOntology() {
    const image = document.getElementById('ontology-image')

    if (image.style.display === 'block') {
        image.style.display = 'none';  // Hide the image if it's currently visible
    } else {
        image.style.display = 'block'; // Show the image if it's currently hidden
    }
}

function fetchTaxonomy() {
    const image = document.getElementById('taxonomy-image');

    if (image.style.display === 'block') {
        image.style.display = 'none';  // Hide the image if it's currently visible
    } else {
        image.style.display = 'block'; // Show the image if it's currently hidden
    }
}

function checkEnter(event) {
    //Check if the pressed key is "Enter"
    if (event.key === "Enter") {
        //Prevent the default form submission if inside a form
        event.preventDefault();

        // Get the user's input from the prompt bar
        const userInput = document.getElementById("prompt-input").value;

        //Call the function to handle the input, passing in the user's text
        //handlePromptInput(userInput);
        console.log(userInput)
        handlePromptInput(userInput);
        //displayGraph();

        //Clear the input field after submission
        document.getElementById("prompt-input").value = '';
    }
}

//Function to handle the user's input
function handlePromptInput(userInput) {
    //alert('DONT TOUCH THAT');
    //

    const chat_response = document.getElementById("chat-responses");
    chat_response.style.display = 'none';

    chat_response.innerHTML = "<h3>You said: </h3>" + userInput + "<br><h3>Llama says: </h3><p><i>Thinking...</i></p>";
    chat_response.style.display = 'block';

    fetch ('/process_input/' , {
        method: 'POST' , 
        headers: {
            'Content-Type' : 'application/json' , 
        } , 
        body: JSON.stringify({prompt: userInput}) ,
    })
    .then(response => response.json())
    .then(data => {
        console.log(data);
        chat_response.innerHTML = `<h3>You said: </h3><br>${userInput}<br><h3>Llama says: </h3><br><p>${data.response}</p>`;
    })
    .catch(error => {
        console.error('Error: ' , error);
        chat_response.innerHTML = `<h3>Error:</h3><br><p>Failed to connect to the server.</p>`;
    });
}

function handleUserInput(userInput) {
    // Send user input to FastAPI backend
    fetch("/process_input", { //specify function within tree_processing to call
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ input_text: userInput })
    })
    .then(response => response.json())
    .then(data => {
        console.log("Output from Python script: ", data.output);
    })
    .catch(error => {
        console.error("Error:", error);
    });
}


//on 'open' of user uploaded pdf store file in /data/raw
document.getElementById("uploadForm").addEventListener("change", function(event) {
    event.preventDefault();  // Prevent form submission
    const form = event.currentTarget;
    const formData = new FormData(form);  // Automatically picks up file input

    // Fetch options with formData for file upload
    const fetchOptions = {
        method: 'POST',
        body: formData,
    };

    fetch("/uploadfile/", fetchOptions)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to upload the file.');
            }
            return response.json();
        })
        .then(data => {
            console.log(data)
            const file_name = document.getElementById("file_name");
            file_name.innerText = `Uploaded File: ${data.filename}`;
            file_name.style.display = 'block';
            alert('File uploaded successfully!');
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error uploading file.');
        });
    console.log("File uploaded successfully")
});



function query_prompt() {
    alert('DONT TOUCH THAT'); // Placeholder
}

// function displayGraph() {
//     const config = {
//       containerId: 'graph',
//       driverUri: 'bolt://localhost:7687',
//       username: 'neo4j',
//       password: 'taxonomies'
//     };

//     const viz = neo4jGraphViz(config);

//     viz.render({
//       //many more queries to come!
//       query: 'MATCH (n)-[r]-(m) RETURN n,r,m'
//     })
// }
function initializeGraph() {
    document.addEventListener("DOMContentLoaded", function () {
        console.log("wtf");
        displayGraph();
    });
}


function displayGraph() {
    const config = {
        container_id: "graph",
        neo4j: {
            initialQuery:`MATCH (n)-[r]->(m) WHERE n.pagerank <> '' RETURN n, r, m;`,
            server_url: "bolt://localhost:7687",
            server_user: "neo4j",
            server_password: "taxonomies",
        },
        labels: {
            "Node": { "caption": "name" }
        },
        relationships: {
            "RELATED": { "caption": false }
        },
    };

    console.log(config.neo4j.initialQuery); // Log the query before passing to NeoVis



    const viz = new NeoVis.default(config);
    viz.render();
}

// function displayGraph() {
//     var config = {
//         container_id: "graph",
//         server_url: "bolt://localhost:7687",
//         server_user: "front-end",
//         server_password: "taxonomy",
//         initial_cypher: "MATCH (n) WHERE n.pagerank IS NOT NULL RETURN n"
//     }


//     var viz = new NeoVis.default(config);
//     viz.render();
// }

function clearChat() {    
    const chat_response = document.getElementById("chat-responses");
    chat_response.innerHTML = "";
    chat_response.style.display = 'none';
}

function copyChatText() {
    var copied_text = document.getElementById("chat-responses");

    // Ensure that the text is selectable
    var text = copied_text.innerText || copied_text.textContent;

    // Copy the text using the Clipboard API
    navigator.clipboard.writeText(text).then(function() {
        alert("Copy that!");
    }).catch(function(error) {
        console.error("Error copying text: ", error);
        alert("Failed to copy text to clipboard.");
    });
}

function regenerateChat() {
    const userInput = document.getElementById("prompt-input").value;
    handleUserInput(userInput);
}