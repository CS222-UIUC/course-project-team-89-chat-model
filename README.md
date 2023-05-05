This is team 89.
### Introduction
ChatModel is a widget that can generate machine learning code. We decided to create this project to assist those who are unfamiliar with software engineering and data science, so that they may also create their own machine learning programs with little experience or prior knowledge necessary.

### Technical Architecture

# Frontend
Navbar: This can direct to any page that we want and it fits in all devices’ displays. It allows users to navigate our site and links to the other components. It is written in JavaScript XML, and I worked on it.

About: This is an introduction of our application and it tells what our application can do and some guidance for users. It precedes the rest of the components and tells users how to use our application.

Demo: This is to show how the customer can use our product to generate the code that they want. We show an image of the page that Sameer built, and it provides a window into the backend components. This is also written in JavaScript XML, and Derrick and I worked on it as well.

Experience: This is how our customers feel after using our product, and provides an outlet so they can share their experiences. It draws in from our codebase’s list of experiences, and works with the feedback component to get user input. As shown in the demo, it currently only has example experiences and not realistic ones. Similar to the other frontend components, this was also implemented in JavaScript XML, and Derrick and I worked on it too.

Feedbacks: This is where users can provide feedback on our app so that we can incorporate them into the design process. It connects to the Experience component as mentioned above and the backend as well, since the feedback has to go somewhere. This is also written in JavaScript XML, and Derrick and I worked on it.

Team: This is the team information and what each member did in this project. It relates to the about segment, as it also details the project in some manner. This is implemented in the same language and Derrick and I worked on it, since these are all part of the React JS application.

# Backend

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
