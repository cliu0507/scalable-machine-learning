package tfidf;

import java.io.*;
import java.util.*;

import org.apache.hadoop.fs.*;
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



public class Prob2TFIDF extends Configured implements Tool{
    public static String dlt = " ";

    public static class Prob2TFIDFMap extends Mapper<LongWritable, Text, Text, Text> {
        public static Text myKey = new Text();
        public static Text myVal = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            StringTokenizer st = new StringTokenizer(value.toString());
            String word = st.nextToken();
            String f_id = st.nextToken();

            StringBuilder builder = new StringBuilder();
            while(st.hasMoreTokens()){
                String filename = st.nextToken();
                String tf_idf = st.nextToken();
                myKey.set(filename);
                builder.append(f_id);
                builder.append(dlt);
                builder.append(word);
                builder.append(dlt);
                builder.append(tf_idf);
                myVal.set(builder.toString());
                context.write(myKey, myVal);
                builder.setLength(0);
            }                               
        }
    }

    public static class Prob2TFIDFReduce extends Reducer<Text, Text, Text, Text> {
        public static Text myVal = new Text();
        public static Text myKey = new Text();

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
          
           Map<Integer, String> myMap = new TreeMap<Integer, String>();
           StringBuilder builder = new StringBuilder();
           for(Text txt : values){
                StringTokenizer st = new StringTokenizer(txt.toString());
                Integer f_id = Integer.parseInt(st.nextToken());
                String word = st.nextToken();
                String tf_idf = st.nextToken();
                builder.append(word);
                builder.append(dlt);
                builder.append(tf_idf);
                myMap.put(f_id, builder.toString());
                builder.setLength(0);
            }

            builder.append("");
            Iterator<Map.Entry<Integer, String> > iterator = myMap.entrySet().iterator();
            while(iterator.hasNext()){
                Map.Entry<Integer, String> mapEntry = iterator.next();
                Integer f_id = mapEntry.getKey();
                StringTokenizer st = new StringTokenizer(mapEntry.getValue());
                st.nextToken();
                String tf_idf = st.nextToken();
                if(iterator.hasNext()){
                    builder.append(f_id);
                    builder.append(":");
                    builder.append(tf_idf);
                    builder.append("\t");
                }else{
                    builder.append(f_id);
                    builder.append(":");
                    builder.append(tf_idf);
                }
            }

            StringTokenizer st = new StringTokenizer(key.toString());
            String filename = st.nextToken();
            String label;
            if(filename.contains("classic90/0") || filename.contains("classic3893/0")){
                label = "1";
            }else if(filename.contains("classic90/1") || filename.contains("classic3893/1")){
                label = "2";
            }else{
                label = "3";
            }
            myKey.set(label);
            myVal.set(builder.toString());
            context.write(myKey, myVal); 
        }
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        conf.setLong("mapreduce.task.timeout", 10000*60*60);

        Job job = new Job(conf, "Prob2TFIDF");
        job.setJarByClass(Prob2TFIDF.class);
      
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setMapperClass(Prob2TFIDF.Prob2TFIDFMap.class);
        job.setReducerClass(Prob2TFIDF.Prob2TFIDFReduce.class);

        job.setNumReduceTasks(300);

        for(int i=0;i<args.length-1;i++){
            FileInputFormat.addInputPath(job, new Path(args[i]));
        }
        FileOutputFormat.setOutputPath(job, new Path(args[args.length-1]));

        return job.waitForCompletion(true)? 0 : 1;
    }

    public static void main(String[] args) throws Exception{
        int res = ToolRunner.run(new Configuration(), new Prob2TFIDF(), args);
        System.exit(res);
    }
}
