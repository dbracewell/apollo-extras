package com.davidbracewell.apollo.mallet.topic;

import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TargetStringToFeatures;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import com.davidbracewell.SystemInfo;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.mallet.VectorToTokensPipe;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.apollo.stat.measure.Similarity;
import com.davidbracewell.guava.common.base.Throwables;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.Setter;

import java.io.IOException;
import java.util.Arrays;

/**
 * @author David B. Bracewell
 */
public class MalletLDALearner extends Clusterer<MalletLDAModel> {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int K = 100;
   @Getter
   @Setter
   private int maxIterations = 2000;
   @Getter
   @Setter
   private int burnIn = 500;
   @Getter
   @Setter
   private int optimizationInterval = 100;


   @Override
   public MalletLDAModel cluster(MStream<NDArray> instances) {
      MalletLDAModel apolloModel = new MalletLDAModel(this, Similarity.Cosine, K);
      apolloModel.pipes = new SerialPipes(Arrays.asList(new TargetStringToFeatures(),
                                                        new VectorToTokensPipe(apolloModel.getFeatureEncoder()),
                                                        new TokenSequence2FeatureSequence()));
      InstanceList trainingData = new InstanceList(apolloModel.pipes);
      instances.filter(i -> i.size() > 0)
               .forEach(
                  i -> trainingData.addThruPipe(
                     new Instance(i, i.getLabel() == null ? "" : i.getLabel().toString(), null, null)));
      apolloModel.topicModel = new ParallelTopicModel(K);
      apolloModel.topicModel.addInstances(trainingData);
      apolloModel.topicModel.setNumIterations(maxIterations);
      apolloModel.topicModel.setNumThreads(SystemInfo.NUMBER_OF_PROCESSORS - 1);
      apolloModel.topicModel.setBurninPeriod(burnIn);
      apolloModel.topicModel.setOptimizeInterval(optimizationInterval);
      apolloModel.topicModel.setSymmetricAlpha(false);
      try {
         apolloModel.topicModel.estimate();
      } catch (IOException e) {
         throw Throwables.propagate(e);
      }

      return apolloModel;
   }

}//END OF MalletLDALearner
