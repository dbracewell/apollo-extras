package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.ClassifierTrainer;
import cc.mallet.classify.DecisionTreeTrainer;
import lombok.Getter;
import lombok.Setter;

/**
 * @author David B. Bracewell
 */
public class DecisionTreeLearner extends MalletClassifierLearner {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int maxDepth = 5;
   @Getter
   @Setter
   private double minInfoGainSplit = 0.001D;

   @Override
   protected ClassifierTrainer<?> getTrainer() {
      DecisionTreeTrainer treeTrainer = new DecisionTreeTrainer();
      treeTrainer.setMaxDepth(maxDepth);
      treeTrainer.setMinInfoGainSplit(minInfoGainSplit);
      return treeTrainer;
   }
}// END OF DecisionTreeLearner
