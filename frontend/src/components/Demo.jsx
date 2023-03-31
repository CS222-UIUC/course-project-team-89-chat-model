import React from 'react'
import { SectionWrapper } from '../loc'
import { styles } from '../styles'
import { demo } from '../assets'

const Demo = () => {
  return (
    <div >
      <div className="m-5">
        <h3 className={styles.sectionHeadText}>Get started</h3>
        <p>Here you can try on our code generator and select the options you want.</p>
        <img src={demo}/>
      </div>

    </div>
  )
}

export default SectionWrapper(Demo, "demo")