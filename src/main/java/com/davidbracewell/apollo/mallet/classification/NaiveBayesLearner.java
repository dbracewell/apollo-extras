package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.NaiveBayesEMTrainer;
import cc.mallet.classify.NaiveBayesTrainer;
import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class NaiveBayesLearner extends MalletClassifierLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private double docLengthNormalization = -1.0;
   @Getter
   @Setter
   private double unLabeledWeight = 0;

   @Override
   protected ClassifierTrainer<?> getTrainer() {
      if (unLabeledWeight > 0) {
         NaiveBayesEMTrainer trainer = new NaiveBayesEMTrainer();
         trainer.setUnlabeledDataWeight(unLabeledWeight);
         trainer.setDocLengthNormalization(docLengthNormalization);
         return trainer;
      } else {
         NaiveBayesTrainer trainer = new NaiveBayesTrainer();
         trainer.setDocLengthNormalization(docLengthNormalization);
         return trainer;
      }
   }

}// END OF NaiveBayesLearner
