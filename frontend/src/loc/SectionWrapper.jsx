
import { styles } from '../styles';

const SectionWrapper = (Component, idName) =>
  function LOC() {
    return (
      <div
        initial="hidden"
        whileInView="show"
        viewport={{ once: true, amount: 0.25 }}
        className={`${styles.padding} max-w-7xl mx-auto relative z-0`}
      >
        <span className='hash-span' id={idName}>
          &nbsp;
        </span>
        <Component />
      </div>
    )
  }

export default SectionWrapper