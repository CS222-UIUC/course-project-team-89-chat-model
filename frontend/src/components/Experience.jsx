import React from 'react'
import { SectionWrapper } from '../loc'
import { styles } from '../styles'

const Experience = () => {
  return (
    <div>
      <div className={styles.heroHeadText}>
        Experience
      </div>
      This will change later
      <li>This app greatly helped me generate my code -- Johnson</li>
      <li>I used this app to improve my efficiency of coding -- Andrew</li>
      <li>This is a very helpful app -- Wilson</li>
    </div>
  )
}

export default SectionWrapper(Experience, "")