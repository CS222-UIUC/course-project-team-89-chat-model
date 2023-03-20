import React from 'react'
import { styles } from '../styles'
import { ai } from '../assets'
import { SectionWrapper } from '../loc';

const About = () => {
  return (
    <>
      <img src={ai} alt="AI Chat Model" className='w-full' />
      <div>
        <h1 className={`${styles.heroHeadText} text-white`}>Hi, It's <span className='text-[#915eff]'>Chat Model</span> </h1>
        <p className={`${styles.heroSubText} mt-2 text-white-100 mb-5`}>
          An AI-based platform, <br className='sm:block hidden' />provides executable classifiers.
        </p>
      </div>
      <p className={styles.sectionSubText}>Introduction</p>
      <p className='mt-4 text-secondary text-[17px] max-w-3xl leading-[30px] mb-5'>
        Nowadays, especially with the introduction of OpenAI’s Chat GPT-3 model into mainstream media,
        artificial intelligence is becoming an even bigger tool used for a wide variety of tasks,
        and those who are unfamiliar with software engineering and data science are being left behind
        by the daunting field, unable to use/understand AI for their own interests. To tackle this
        issue, we propose building a highly interactive website in which users can specify dimensions
        and parameters to a neural network, along with providing a dataset to train their model on.
        Our product aims to save the trained weights of the model to provide an executable
        classifier/regressor along with generated code for the AI based on the specifications.
        Ideally, our software would decrease the difficulty of implementing machine learning,
        much as Scratch decreased the difficulty of algorithmic thinking.

      </p>
    </>
  )
}

export default SectionWrapper(About, "about")