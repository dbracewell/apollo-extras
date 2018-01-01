package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.types.Labeling;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;

/**
 * @author David B. Bracewell
 */
public class MalletClassifier extends Classifier {
   private static final long serialVersionUID = 1L;
   cc.mallet.classify.Classifier model;

   protected MalletClassifier(ClassifierLearner learner) {
      super(learner);
   }


   @Override
   public Classification classify(NDArray vector) {
      Labeling labeling = model.classify(model.getInstancePipe()
                                              .instanceFrom(new cc.mallet.types.Instance(vector, "", null, null)))
                               .getLabeling();
      double[] result = new double[getLabelEncoder().size()];
      for (int i = 0; i < getLabelEncoder().size(); i++) {
         result[i] = labeling.value(i);
      }
      return createResult(result);
   }

}// END OF MalletClassifier
