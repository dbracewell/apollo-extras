package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.AdaBoostTrainer;
import cc.mallet.classify.DecisionTreeTrainer;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.InstanceList;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;

/**
 * @author David B. Bracewell
 */
public class AdaBoostLearner extends MalletClassifierLearner {
   @Override
   protected Classifier trainInstanceList(InstanceList instances, Pipe pipe, PreprocessorList<Instance> preprocessors) {
      MalletClassifier classifier = new MalletClassifier(instances.getTargetAlphabet(), instances.getDataAlphabet(), preprocessors);
      AdaBoostTrainer trainer = new AdaBoostTrainer(new DecisionTreeTrainer(),10);
      classifier.model = trainer.train(instances);
      return classifier;
   }
}// END OF AdaBoostLearner
