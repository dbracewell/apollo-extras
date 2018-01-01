package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.C45Trainer;
import cc.mallet.classify.ClassifierTrainer;
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
   protected ClassifierTrainer<?> getTrainer() {
      C45Trainer trainer = new C45Trainer();
      trainer.setDepthLimited(depthLimited);
      trainer.setDoPruning(doPruning);
      trainer.setMaxDepth(maxDepth);
      trainer.setMinNumInsts(minInstances);
      return trainer;
   }

}// END OF C45Learner
