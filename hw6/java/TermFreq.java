package tfidf;

import java.io.IOException;
import java.util.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.hadoop.fs.Path;
import org.apache.hadoop.conf.*;
import org.apache.hadoop.io.*;
import org.apache.hadoop.util.*;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.input.FileSplit;

public class TermFreq extends Configured implements Tool{
    public static String dlt = " ";

    public static class TermFreqMap extends Mapper<LongWritable, Text, Text, IntWritable> {
        private static Set<String> googleStopwords;
        //stop words copied from web
        static{
            googleStopwords = new HashSet<String>();
            googleStopwords.add("I"); googleStopwords.add("a"); googleStopwords.add("about");
            googleStopwords.add("an"); googleStopwords.add("are"); googleStopwords.add("as");
            googleStopwords.add("at"); googleStopwords.add("be"); googleStopwords.add("by");
            googleStopwords.add("com"); googleStopwords.add("de"); googleStopwords.add("en");
            googleStopwords.add("for"); googleStopwords.add("from"); googleStopwords.add("how");
            googleStopwords.add("in"); googleStopwords.add("is"); googleStopwords.add("it");
            googleStopwords.add("la"); googleStopwords.add("of"); googleStopwords.add("on");
            googleStopwords.add("or"); googleStopwords.add("that"); googleStopwords.add("the");
            googleStopwords.add("this"); googleStopwords.add("to"); googleStopwords.add("was");
            googleStopwords.add("what"); googleStopwords.add("when"); googleStopwords.add("where"); 
            googleStopwords.add("who"); googleStopwords.add("will"); googleStopwords.add("with");
            googleStopwords.add("and"); googleStopwords.add("the"); googleStopwords.add("www");
        }

        private static final Pattern PATTERN = Pattern.compile("\\w+");
        private Text word = new Text();
        private final static IntWritable one = new IntWritable(1);

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            //This map code is similar to tutorial online by removing stop words
            Matcher m = PATTERN.matcher(value.toString());            

            String filename = ((FileSplit) context.getInputSplit()).getPath().toString();
            StringBuilder builder = new StringBuilder();
            while(m.find()){
                String matchedKey = m.group().toLowerCase();
                if(!Character.isLetter(matchedKey.charAt(0)) || Character.isDigit(matchedKey.charAt(0)) || googleStopwords.contains(matchedKey) || matchedKey.contains("_") || matchedKey.length() < 3){
                    continue;
                }
                builder.append(filename);
                builder.append(dlt);
                builder.append(matchedKey);
                word.set(builder.toString());
                builder.setLength(0);
                context.write(word, one);
            }
        }
    }

    public static class TermFreqReduce extends Reducer<Text, IntWritable, Text, Text> {
        private Text myKey = new Text();
        private Text myVal = new Text();

        public void reduce(Text key, Iterable<IntWritable> values, Context context) throws IOException, InterruptedException{
            int sum = 0;
            for(IntWritable val : values) {
                sum += val.get();
            }
            String[] pair = key.toString().split(dlt);
            String filename = pair[0];
            String word = pair[1];
            myKey.set(filename);
            StringBuilder builder = new StringBuilder();
            builder.append(pair[1]);
            builder.append(dlt);
            builder.append(sum);
            myVal.set(builder.toString());
            context.write(myKey, myVal);
        }
    }

    public int run(String[] args) throws Exception {
        Job job = new Job(getConf(), "TermFreq");
        job.setJarByClass(TermFreq.class);
      
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(IntWritable.class);

        job.setMapperClass(TermFreq.TermFreqMap.class);
        job.setReducerClass(TermFreq.TermFreqReduce.class);

        job.setNumReduceTasks(300);

        for(int i=0;i<args.length-1;i++){
            FileInputFormat.addInputPath(job, new Path(args[i]));
        }
        FileOutputFormat.setOutputPath(job, new Path(args[args.length-1]));

        return job.waitForCompletion(true)? 0 : 1;
    }

    public static void main(String[] args) throws Exception{
        int res = ToolRunner.run(new Configuration(), new TermFreq(), args);
        System.exit(res);
    }
}
