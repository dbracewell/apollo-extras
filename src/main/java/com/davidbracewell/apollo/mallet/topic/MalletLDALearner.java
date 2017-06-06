package com.davidbracewell.apollo.mallet.topic;

import cc.mallet.pipe.SerialPipes;
import cc.mallet.pipe.TargetStringToFeatures;
import cc.mallet.pipe.TokenSequence2FeatureSequence;
import cc.mallet.topics.ParallelTopicModel;
import cc.mallet.types.Alphabet;
import cc.mallet.types.IDSorter;
import cc.mallet.types.Instance;
import cc.mallet.types.InstanceList;
import com.davidbracewell.SystemInfo;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.mallet.VectorToTokensPipe;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.collection.counter.Counter;
import com.davidbracewell.collection.counter.Counters;
import com.davidbracewell.guava.common.base.Throwables;
import com.davidbracewell.stream.MStream;
import lombok.Getter;
import lombok.Setter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.TreeSet;

/**
 * @author David B. Bracewell
 */
public class MalletLDALearner extends Clusterer<MalletLDAModel> {
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

   private Counter<String> getTopicWords(int topic, ParallelTopicModel topicModel, SerialPipes pipes) {
      final Alphabet alphabet = pipes.getDataAlphabet();
      final ArrayList<TreeSet<IDSorter>> topicWords = topicModel
                                                         .getSortedWords();
      double[][] termScores = topicModel
                                 .getTopicWords(
                                    true,
                                    true);
      Iterator iterator = topicWords
                             .get(
                                topic)
                             .iterator();
      IDSorter info;
      Counter<String> topicWordScores = Counters
                                           .newCounter();
      while (iterator.hasNext()) {
         info = (IDSorter) iterator.next();
         topicWordScores.set(
            alphabet
               .lookupObject(
                  info.getID())
               .toString(),
            termScores[topic][info.getID()]);
      }
      return topicWordScores;
   }


   @Override
   public MalletLDAModel cluster(MStream<Vector> instances) {
      MalletLDAModel apolloModel = new MalletLDAModel();
      apolloModel.pipes = new SerialPipes(Arrays.asList(new TargetStringToFeatures(),
                                                        new VectorToTokensPipe(),
                                                        new TokenSequence2FeatureSequence()));
      InstanceList trainingData = new InstanceList(apolloModel.pipes);
      instances.filter(i -> i.size() > 0)
               .forEach(i -> trainingData.addThruPipe(new Instance(i, "", null, null)));
      apolloModel.topicModel = new ParallelTopicModel(K);
      apolloModel.topicModel.addInstances(trainingData);
      apolloModel.topicModel.setNumIterations(maxIterations);
      apolloModel.topicModel.setNumThreads(SystemInfo.NUMBER_OF_PROCESSORS - 1);
      apolloModel.topicModel.setBurninPeriod(burnIn);
      apolloModel.topicModel.setOptimizeInterval(optimizationInterval);
      try {
         apolloModel.topicModel.estimate();
      } catch (IOException e) {
         throw Throwables.propagate(e);
      }

      for (int i = 0; i < K; i++) {
         System.out.println(getTopicWords(i, apolloModel.topicModel, apolloModel.pipes).topN(5).itemsByCount(false));
      }

      return apolloModel;
   }

}//END OF MalletLDALearner
