import { BrowserRouter } from "react-router-dom";
import { Navbar, Demo, About, Contact, Experience, Feedbacks } from './components';
import Teams from "./components/Teams";

const App = () => {
  return (
    <BrowserRouter>
      <div className="relative z-0 bg-primary">
        <div className='bg-hero-pattern bg-cover bg-no-repeat bg-center'>
          <Navbar />
          <Demo />
        </div>
        <About />
        <Experience />
        <Teams />
        <Feedbacks />
        <div className="relative z-0">

          <Contact />
        </div>
      </div>
    </BrowserRouter>
  )
}

export default App
