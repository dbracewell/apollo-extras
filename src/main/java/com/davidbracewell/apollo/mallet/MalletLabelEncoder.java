package com.davidbracewell.apollo.mallet;

import cc.mallet.types.Alphabet;
import com.davidbracewell.apollo.ml.Example;
import com.davidbracewell.apollo.ml.LabelEncoder;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.stream.MStream;

import java.io.Serializable;
import java.util.Arrays;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class MalletLabelEncoder implements LabelEncoder, Serializable {
   private static final long serialVersionUID = 1L;
   private final Alphabet labelAlphabet;

   public MalletLabelEncoder(Alphabet labelAlphabet) {
      this.labelAlphabet = labelAlphabet;
   }

   @Override
   public LabelEncoder createNew() {
      return new MalletLabelEncoder(labelAlphabet);
   }

   @Override
   public Object decode(double value) {
      return labelAlphabet.lookupObject((int) value);
   }

   @Override
   public double encode(Object object) {
      return labelAlphabet.lookupIndex(object);
   }

   @Override
   public void fit(Dataset<? extends Example> dataset) {

   }

   @Override
   public void fit(MStream<String> stream) {

   }

   @Override
   public void freeze() {
      labelAlphabet.stopGrowth();
   }

   @Override
   public double get(Object object) {
      return labelAlphabet.lookupIndex(object);
   }

   @Override
   public boolean isFrozen() {
      return labelAlphabet.growthStopped();
   }

   @Override
   public int size() {
      return labelAlphabet.size();
   }

   @Override
   public void unFreeze() {
      labelAlphabet.startGrowth();
   }

   @Override
   public List<Object> values() {
      return Arrays.asList(labelAlphabet.toArray());
   }
}// END OF MalletLabelEncoder
