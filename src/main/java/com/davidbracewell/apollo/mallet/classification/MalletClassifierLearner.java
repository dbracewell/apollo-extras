package com.davidbracewell.apollo.mallet.classification;

import cc.mallet.pipe.*;
import cc.mallet.types.InstanceList;
import com.davidbracewell.apollo.mallet.VectorToTokensPipe;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.classification.Classifier;
import com.davidbracewell.apollo.ml.classification.ClassifierLearner;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;

import java.util.Arrays;

/**
 * @author David B. Bracewell
 */
public abstract class MalletClassifierLearner extends ClassifierLearner {
   private static final long serialVersionUID = 1L;

   protected SerialPipes createPipe() {
      return new SerialPipes(Arrays.asList(new Target2Label(),
                                           new VectorToTokensPipe(),
                                           new TokenSequence2FeatureSequence(),
                                           new FeatureSequence2FeatureVector()));
   }

   @Override
   public void reset() {

   }

   @Override
   protected Classifier trainImpl(Dataset<Instance> dataset) {
      Pipe pipe = createPipe();
      InstanceList trainingData = new InstanceList(pipe);
      dataset.asFeatureVectors()
             .forEach(i -> {
                String lbl = dataset.getEncoderPair().decodeLabel(i.getLabel()).toString();
                trainingData.addThruPipe(new cc.mallet.types.Instance(i, lbl, null, null));
             });
      return trainInstanceList(trainingData, pipe, dataset.getPreprocessors());
   }

   protected abstract Classifier trainInstanceList(InstanceList instances, Pipe pipe, PreprocessorList<Instance> preprocessors);

}// END OF MalletClassifierLearner
