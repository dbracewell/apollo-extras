package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.C45Trainer;
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
public class C45Learner extends MalletClassifierLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private boolean depthLimited = false;
   @Getter
   @Setter
   private boolean doPruning = true;
   @Getter
   @Setter
   private int maxDepth = 4;
   @Getter
   @Setter
   private int minInstances = 2;

   @Override
   protected Classifier trainInstanceList(InstanceList instances, Pipe pipe, PreprocessorList<Instance> preprocessors) {
      C45Trainer trainer = new C45Trainer();
      trainer.setDepthLimited(depthLimited);
      trainer.setDoPruning(doPruning);
      trainer.setMaxDepth(maxDepth);
      trainer.setMinNumInsts(minInstances);
      MalletClassifier model = new MalletClassifier(instances.getTargetAlphabet(),
                                                    instances.getDataAlphabet(),
                                                    preprocessors);
      model.model = trainer.train(instances);
      return model;
   }
}// END OF C45Learner
