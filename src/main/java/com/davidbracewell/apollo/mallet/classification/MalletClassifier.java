package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.types.Alphabet;
import cc.mallet.types.Labeling;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.mallet.MalletFeatureEncoder;
import com.davidbracewell.apollo.mallet.MalletLabelEncoder;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classification;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;

/**
 * @author David B. Bracewell
 */
public abstract class MalletClassifier extends Classifier {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Classifier.
    *
    * @param targetAlphabet  the target alphabet
    * @param featureAlphabet the feature alphabet
    * @param preprocessors   the preprocessors that the classifier will need apply at runtime
    */
   protected MalletClassifier(Alphabet targetAlphabet, Alphabet featureAlphabet, PreprocessorList<Instance> preprocessors) {
      super(new EncoderPair(new MalletLabelEncoder(targetAlphabet), new MalletFeatureEncoder(featureAlphabet)),
            preprocessors);
   }

   @Override
   public Classification classify(Vector vector) {
      cc.mallet.classify.Classifier model = getClassifier();
      Labeling labeling = model.classify(model.getInstancePipe()
                                              .instanceFrom(new cc.mallet.types.Instance(vector, "", null, null)))
                               .getLabeling();
      double[] result = new double[getLabelEncoder().size()];
      for (int i = 0; i < getLabelEncoder().size(); i++) {
         result[i] = labeling.value(i);
      }
      return createResult(result);
   }

   protected abstract cc.mallet.classify.Classifier getClassifier();
}// END OF MalletClassifier
