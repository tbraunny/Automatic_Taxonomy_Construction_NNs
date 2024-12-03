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

function clearChat() {
    alert("DONT TOUCH THAT");
}

function copyChatText() {
    alert("I SAID DONT TOUCH THAT")
}

function regenerateChat() {
    alert("STOP IT")
}

// function display_about() {
//     const aboutSection = document.getElementById('about-section');

//     const layerDefinitionsBox = document.getElementById('responses');
//     layerDefinitionsBox.style.display = 'none';

//     aboutSection.innerHTML = `<h3>About / Open Source</h3>
//         <p>This project aims to provide automatic taxonomy construction for neural networks.<br>
//         It categorizes layers, architectures, and loss functions based on input given from the user.<br>
//         We are a 4-person team from the University of Nevada, Reno.<br>
//         Team Consists of:<br>
//         -Thomas Braun<br>
//         -Lukas Lac <br>
//         -Josue Ochoa <br>
//         -Richard White <br> 

//         All content is open-source and welcomes contributions.</p>`;

//         if (aboutSection.style.display === 'block') {
//             aboutSection.style.display = 'none';  // Hide the image if it's currently visible
//         } else {
//             aboutSection.style.display = 'block'; // Show the image if it's currently hidden
//         }
// }

function checkEnter(event) {
    //Check if the pressed key is "Enter"
    if (event.key === "Enter") {
        //Prevent the default form submission if inside a form
        event.preventDefault();

        // Get the user's input from the prompt bar
        const userInput = document.getElementById("prompt-input").value;

        //Call the function to handle the input, passing in the user's text
        handlePromptInput(userInput);

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

    chat_response.innerHTML = "<h3>You said: </h3><br>" + userInput + "<br><h3>Llama says: </h3><br><p>My developers are still working on me<br>I will be answering all of your questions soon!</p>";
    chat_response.style.display = 'block';

    fetch ('/api/chat' , {
        method: 'POST' , 
        headers: {
            'Content-Type' : 'application/json' , 
        } , 
        body: JSON.stringify({prompt: userInput}) ,
    })
    .then(response => response.json())
    .then(data => {
        chat_response.innerHTML = `<h3>You said: </h3><br>${userInput}<br><h3>Llama says: </h3><br><p>${data.response}</p>`;
    })
    .catch(error => {
        console.error('Error: ' , error);
        chat_response.innerHTML = `<h3>Error:</h3><br><p>Failed to connect to the server.</p>`;
    });
}

const form = document.querySelector('form');
form.addEventListener('submit', uploadFile);

/** @param {Event} event */
function uploadFile(event) {
    //alert('DONT TOUCH THAT'); // Placeholder action for file upload button
    const form = event.currentTarget;
    const url = new URL(form.action);
    const formData = new FormData(form)
    const searchParams = new URLSearchParams(formData);

    /** @type {Parameters<fetch>[1]} */
    const fetchOptions = {
        method: form.method,
    };

    if (form.method.toLowerCase() === 'post') {
        if (form.enctype === 'multipart/form-data') {
            fetchOptions.body = formData;
        }
        else {
            fetchOptions.body = searchParams;
        }
    }
    else {
        url.search = searchParams;
    }            
    
    fetch(url , fetchOptions);
    event.preventDefault();
}



function query_prompt() {
    alert('DONT TOUCH THAT'); // Placeholder
}