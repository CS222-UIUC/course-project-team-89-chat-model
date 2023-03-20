import React from 'react'
import { SectionWrapper } from '../loc'
import { styles } from '../styles'



const Teams = () => {
  return (
    <div>
      <h3 className={styles.sectionHeadText}>Teams Members</h3>
      <p>
        Our group’s primary mode of communication is through a discord channel and email.
        Given that we do not often meet up and instead communicate online, we will stay
        together by meeting on a platform like Zoom every week to check in with each other.
        At each team meeting, we will analyze the work remaining and divide up the action
        items by seeing whose skills fit best. For example, Jiayuan has strong React and
        frontend experience, and will likely take frontend-related tasks, while Sameer has
        more experience with the backend and machine learning.
      </p>
    </div>
  )
}

export default SectionWrapper(Teams, "team");