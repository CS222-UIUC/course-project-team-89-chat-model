import React from 'react'
import { styles } from '../styles'
import { ai } from '../assets'
import { SectionWrapper } from '../loc';

const About = () => {
  return (
    <>
      <img src={ai} alt="AI Chat Model" className='w-full' />
      <div>
        <h1 className={`${styles.heroHeadText} text-white`}>The future of AI <span className='text-[#915eff]'>Chat Models</span> </h1>
        <p className={`${styles.heroSubText} mt-2 text-white-100 mb-5`}>
          An AI-run service that <br className='sm:block hidden' />provides executable models.
        </p>
      </div>
      <p className={styles.sectionSubText}>Introduction</p>
      <p className='mt-4 text-secondary text-[17px] max-w-3xl leading-[30px] mb-5'>
        In the modern era, artificial intelligence and machine learning are becoming larger and more popular
        tools for a variety of tasks. In particular, the introduction of OpenAI's ChatGPT has shown the
        possibilities of modern machine learning models. We fear that those who are unfamiliar with software
        engineering and data science will be left behind by this advancing era, unable to adapt to using AI
        or even understand how it works.

        To tackle this issue, our website aims to provide users an easy way to create machine learning code.
        Users can specify dimensions and parameters to a neural network and provide a dataset for the AI to
        train their model on. Our site will generate the code for a basic AI model based on the inputs the
        user provided.
      </p>
    </>
  )
}

export default SectionWrapper(About, "about")