package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.classify.Classifier;
import cc.mallet.types.Alphabet;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.io.Resources;
import com.davidbracewell.io.resource.Resource;

import java.util.Random;

import static com.davidbracewell.apollo.ml.classification.ClassifierEvaluation.crossValidation;

/**
 * The type C 45.
 *
 * @author David B. Bracewell
 */
public class C45 extends MalletClassifier {
   private static final long serialVersionUID = 1L;
   /**
    * The Model.
    */
   cc.mallet.classify.C45 model;

   /**
    * Instantiates a new Classifier.
    *
    * @param targetAlphabet  the target alphabet
    * @param featureAlphabet the feature alphabet
    * @param preprocessors   the preprocessors that the classifier will need apply at runtime
    */
   protected C45(Alphabet targetAlphabet, Alphabet featureAlphabet, PreprocessorList<Instance> preprocessors) {
      super(targetAlphabet, featureAlphabet, preprocessors);
   }

   public static void main(String[] args) {
      Resource url = Resources.from(
         "https://raw.githubusercontent.com/sjwhitworth/golearn/master/examples/datasets/iris_headers.csv");
      DenseCSVDataSource dataSource = new DenseCSVDataSource(url, true);
      dataSource.setLabelName("Species");
      Dataset<Instance> dataset = Dataset.classification()
                                         .source(dataSource)
                                         .shuffle(new Random(1234));
      crossValidation(dataset,
                      () -> new NaiveBayesLearner(),
                      10
                     )
         .output(System.out);
   }

   @Override
   protected Classifier getClassifier() {
      return model;
   }

}// END OF C45
