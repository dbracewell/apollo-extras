package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.types.Alphabet;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;

/**
 * @author David B. Bracewell
 */
public class NaiveBayes extends MalletClassifier {
   private static final long serialVersionUID = 1L;
   cc.mallet.classify.NaiveBayes model;

   /**
    * Instantiates a new Classifier.
    *
    * @param targetAlphabet  the target alphabet
    * @param featureAlphabet the feature alphabet
    * @param preprocessors   the preprocessors that the classifier will need apply at runtime
    */
   protected NaiveBayes(Alphabet targetAlphabet, Alphabet featureAlphabet, PreprocessorList<Instance> preprocessors) {
      super(targetAlphabet, featureAlphabet, preprocessors);
   }

   @Override
   protected cc.mallet.classify.Classifier getClassifier() {
      return model;
   }

}// END OF NaiveBayes
