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

public class ReTFIDF extends Configured implements Tool{
    public static String dlt = " ";

    public static class ReTFIDFMap extends Mapper<LongWritable, Text, Text, Text> { 
        private Text myKey = new Text();
        private Text myVal = new Text();
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{

            StringTokenizer st = new StringTokenizer(value.toString());
            String word = st.nextToken();
            String cur_id = st.nextToken();
            String filename = st.nextToken();
            String tf_idf = st.nextToken();

            StringBuilder builder = new StringBuilder();
            builder.append(word);
            builder.append(dlt);
            builder.append(cur_id);
            myKey.set(builder.toString());
            builder.setLength(0);

            builder.append(filename);
            builder.append(dlt);
            builder.append(tf_idf);
            myVal.set(builder.toString());
            context.write(myKey, myVal);
        }
    }

    public static class ReTFIDFReduce extends Reducer<Text, Text, Text, Text> {
        private Text myVal = new Text();
        private Text myKey = new Text();

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{

            StringTokenizer st = new StringTokenizer(key.toString());
            String word = st.nextToken();
            String f_id = st.nextToken();
            
            StringBuilder builder = new StringBuilder(f_id);
            builder.append(dlt);
            for(Text txt : values){
                builder.append(txt.toString());
                builder.append("\t");
            }
            myKey.set(word);
            myVal.set(builder.toString());
            context.write(myKey, myVal);
        }
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();

        //set corpus size
        conf.setLong("mapreduce.task.timeout", 10000*60*60);   
 
        Job job = new Job(conf, "ReTFIDF");
        job.setJarByClass(ReTFIDF.class);
      
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setMapperClass(ReTFIDF.ReTFIDFMap.class);
        job.setReducerClass(ReTFIDF.ReTFIDFReduce.class);

        job.setNumReduceTasks(300);

        for(int i=0;i<args.length-1;i++){
            FileInputFormat.addInputPath(job, new Path(args[i]));
        }
        FileOutputFormat.setOutputPath(job, new Path(args[args.length-1]));
        
        return job.waitForCompletion(true)? 0 : 1;
    }

    public static void main(String[] args) throws Exception{
        int res = ToolRunner.run(new Configuration(), new ReTFIDF(), args);
        System.exit(res);
    }
}
