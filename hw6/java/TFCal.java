package tfidf;

import java.io.IOException;
import java.util.List;
import java.util.ArrayList;

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



public class TFCal extends Configured implements Tool{
    public static String dlt = " ";

    public static class TFCalMap extends Mapper<LongWritable, Text, Text, Text> {
        private Text myKey = new Text();
        private Text myVal = new Text();  
 
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            String[] pair = value.toString().split("\t");
            myKey.set(pair[0]); 
            myVal.set(pair[1]);
            context.write(myKey, myVal);
        }
    }
    public static class TFCalReduce extends Reducer<Text, Text, Text, Text>{
        private Text myVal = new Text();

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
            List<String> cur_list = new ArrayList<String>();            
            
            int norm_cnt = 0;
            //get normalize constants
            for(Text txt : values){
                String cur_str = txt.toString();
                cur_list.add(cur_str);
                norm_cnt += Integer.parseInt(cur_str.split(dlt)[1]);
            }        

            StringBuilder builder = new StringBuilder("");
            //calculate TF for each word, and collect result
            for(String str : cur_list){
                String[] pair = str.split(dlt);
                String word = pair[0];
                double word_cnt = Double.parseDouble(pair[1]);
                double tf_word = word_cnt/norm_cnt;
                builder.append(word);
                builder.append(dlt);
                builder.append(tf_word);
                builder.append("\t");
            }

            myVal.set(builder.toString());
            context.write(key, myVal);
        }
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        conf.setLong("mapreduce.task.timeout",10000*60*60);

        Job job = new Job(conf, "TFCal");
        job.setJarByClass(TFCal.class);
      
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setMapperClass(TFCal.TFCalMap.class);
        job.setReducerClass(TFCal.TFCalReduce.class);
        
        job.setNumReduceTasks(300);

        for(int i=0;i<args.length-1;i++){
            FileInputFormat.addInputPath(job, new Path(args[i]));
        }
        FileOutputFormat.setOutputPath(job, new Path(args[args.length-1]));

        return job.waitForCompletion(true)? 0 : 1;
    }

    public static void main(String[] args) throws Exception{
        int res = ToolRunner.run(new Configuration(), new TFCal(), args);
        System.exit(res);
    }
}
