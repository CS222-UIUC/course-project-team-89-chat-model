import { useState, useRef } from 'react'
import { styles } from '../styles'
import emailjs from '@emailjs/browser';
import { SectionWrapper } from '../loc';

const Contact = () => {
  const formRef = useRef();
  const [form, setForm] = useState({
    name: '',
    email: '',
    message: ''
  })
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm({ ...form, [name]: value })
  }

  const handleSubmit = (e) => {
    e.preventDefault();
    setLoading(true);

    emailjs.send(
      'service_7fkueyi',
      'template_y4p92kq',
      {
        from_name: form.name,
        to_name: 'Albert',
        to_email: 'einsgatehong@gmail.com',
        message: form.message
      },
      'mMKrZWTaP46w8JQ3y'
    )
      .then(() => {
        setLoading(false);
        alert('Received your message! We will call you back ASAP!');

        setForm({
          name: '',
          email: '',
          message: ""
        })
      }, (error) => {
        setLoading(false);
        console.log(error);
        alert("Something wrong");
      })
  }

  return (
    <div>
      <p className={styles.sectionSubText}>Get in Touch</p>
      <h3 className={styles.sectionHeadText}>Contact.</h3>
      <form
        ref={formRef}
        onSubmit={handleSubmit}
        className="mt-12 flex flex-col gap-8"
      >
        <label className="flex flex-col">
          <span className="text-white font-medium mb-4">Your Name</span>
          <input
            type="text"
            name="name"
            value={form.name}
            onChange={handleChange}
            placeholder="Enter your name"
            className="bg-tertiary py-4 px-6 placeholder:text-secondary
               text-white rounded-lg outline-none border-none font-medium"
          />
        </label>
        <label className="flex flex-col">
          <span className="text-white font-medium mb-4">Your Email</span>
          <input
            type="email"
            name="email"
            value={form.email}
            onChange={handleChange}
            placeholder="Enter your email"
            className="bg-tertiary py-4 px-6 placeholder:text-secondary
               text-white rounded-lg outline-none border-none font-medium"
          />
        </label>
        <label className="flex flex-col">
          <span className="text-white font-medium mb-4">Your Message</span>
          <textarea
            rows="7"

            name="message"
            value={form.message}
            onChange={handleChange}
            placeholder="Enter your message"
            className="bg-tertiary py-4 px-6 placeholder:text-secondary
               text-white rounded-lg outline-none border-none font-medium"
          />
        </label>
        <button
          type="submit"
          className="py-3 bg-tertiary px-8 outline-none w-fit
           text-white font-bold shadow-md shadow-primary rounded-xl"
        >
          {loading ? 'Sending...' : 'Send'}
        </button>
      </form>
    </div>
  )
}

export default SectionWrapper(Contact, "contact")