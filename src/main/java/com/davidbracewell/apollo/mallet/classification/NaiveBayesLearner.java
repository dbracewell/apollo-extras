package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.NaiveBayesTrainer;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.InstanceList;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;

/**
 * @author David B. Bracewell
 */
public class NaiveBayesLearner extends MalletClassifierLearner {
   private static final long serialVersionUID = 1L;
   @Override
   protected Classifier trainInstanceList(InstanceList instances, Pipe pipe, PreprocessorList<Instance> preprocessors) {
      NaiveBayes nb = new NaiveBayes(instances.getTargetAlphabet(), instances.getDataAlphabet(), preprocessors);
      NaiveBayesTrainer trainer = new NaiveBayesTrainer();
      nb.model = trainer.train(instances);
      return nb;
   }
}// END OF NaiveBayesLearner
