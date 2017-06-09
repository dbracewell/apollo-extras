package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.AdaBoostTrainer;
import cc.mallet.classify.ClassifierTrainer;
import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class AdaBoostLearner extends MalletClassifierLearner {
   @Getter
   @Setter
   private int numberOfRounds = 100;
   @Getter
   @Setter
   private MalletClassifierLearner weakLearner = new DecisionTreeLearner();


   @Override
   protected ClassifierTrainer<?> getTrainer() {
      return new AdaBoostTrainer(weakLearner.getTrainer(), numberOfRounds);
   }


}// END OF AdaBoostLearner
