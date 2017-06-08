package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.NaiveBayesEMTrainer;
import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.InstanceList;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
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
   protected Classifier trainInstanceList(InstanceList instances, Pipe pipe, PreprocessorList<Instance> preprocessors) {
      MalletClassifier nb = new MalletClassifier(instances.getTargetAlphabet(), instances.getDataAlphabet(),
                                                 preprocessors);
      if (unLabeledWeight > 0) {
         NaiveBayesEMTrainer trainer = new NaiveBayesEMTrainer();
         trainer.setUnlabeledDataWeight(unLabeledWeight);
         trainer.setDocLengthNormalization(docLengthNormalization);
         nb.model = trainer.train(instances);
      } else {
         NaiveBayesTrainer trainer = new NaiveBayesTrainer();
         trainer.setDocLengthNormalization(docLengthNormalization);
         nb.model = trainer.train(instances);
      }
      return nb;
   }
}// END OF NaiveBayesLearner
