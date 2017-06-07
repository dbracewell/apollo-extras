package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.C45Trainer;
import cc.mallet.pipe.Pipe;
import cc.mallet.types.InstanceList;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;

/**
 * @author David B. Bracewell
 */
public class C45Learner extends MalletClassifierLearner {
   private static final long serialVersionUID = 1L;

   @Override
   protected Classifier trainInstanceList(InstanceList instances, Pipe pipe, PreprocessorList<Instance> preprocessors) {
      C45Trainer trainer = new C45Trainer();
      C45 model = new C45(instances.getTargetAlphabet(), instances.getDataAlphabet(), preprocessors);
      model.model = trainer.train(instances);
      return model;
   }
}// END OF C45Learner
