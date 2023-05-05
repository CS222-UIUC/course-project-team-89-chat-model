This is team 89.
### Introduction
ChatModel is a widget that can generate machine learning code. We decided to create this project to assist those who are unfamiliar with software engineering and data science, so that they may also create their own machine learning programs with little experience or prior knowledge necessary.

### Technical Architecture
The backend is created using HTML, C#, CSS, asp.net, and Razor. It accepts user parameters for their desired neural network. The user is also given default parameters if they are unsure on what parameters to give. 

The database stores boilerplate code files and was written in Python and asp.net. The code is dynamically modified and returned based on the user's input.

The Code Generator uses an MLFileWriter class that reads from the database's files and outputs updated code. It was created primarily in C# and Razor.

The final output is generated in a large blue box for the user to view. The field can be copied and pasted elsewhere, and it is still possible to update the parameters of the code used to generate it.

The frontend is created using ReactJS. It was created to be appealing and easy to use for those with no software engineering knowledge.

### Installation Instructions
To use our application, you need to download the zipped file containing our code. To view our frontend landing page and use functionality like contacting us, you would need to run "npm install" and "npm run dev". After this, click on the link in the terminal and that will take you to the home page.

To use the code generator widget, you would need to navigate into the backend/WebApp directory and run dotnet watch. This will open up the widget, and you can enter parameters or view the defaults like in the demo.


### Group Members:
__Jiayuan (Albert) Hong:__ Main coder of the frontend portion of the project. Worked on the GUI.

__Sameer Komoravolu:__ Main coder of the backend portion of the project. Coded the code generator.

__Derrick Kim:__ Assistant coder; provided support to front or backend where it was needed.
