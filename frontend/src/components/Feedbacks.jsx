import React from 'react'
import { SectionWrapper } from '../loc'
import { styles } from '../styles'
const Feedbacks = () => {
  return (
    <div>
      <div className={styles.heroHeadText}>
        Feedback
      </div>
      <li>This app greatly helped me generate my code -- Johnson</li>
      <li>I used this app to improve my efficiency of coding -- Andrew</li>
      <li>This is a very helpful app -- Wilson</li>
    </div>
  )
}

export default SectionWrapper(Feedbacks,"")