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

public class TFIDFCal extends Configured implements Tool{
    public static String dlt = " ";

    public static class TFIDFCalMap extends Mapper<LongWritable, Text, Text, Text> { 
        private Text myKey = new Text();
        private Text myVal = new Text();
        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{

            StringTokenizer st = new StringTokenizer(value.toString());
            String filename = st.nextToken();

            StringBuilder builder = new StringBuilder();
            while(st.hasMoreTokens()){
                String word = st.nextToken();
                String word_tf = st.nextToken();
                myKey.set(word);
                builder.append(filename);
                builder.append(dlt);
                builder.append(word_tf);

                myVal.set(builder.toString());
                context.write(myKey, myVal);
                builder.setLength(0);
            }
        }
    }

    public static class TFIDFCalReduce extends Reducer<Text, Text, Text, Text> {
        private Text myVal = new Text();
        private static int cur_id = 0;

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
            cur_id++;

            ArrayList<String> cur_list = new ArrayList<String>();
            int corpus_size = 90;
            //int corpus_size = 3893;
            //int corpus_size = 300000;
            //int corpus_size = 50000;

            //get word in doc count
            int in_doc_cnt = 0;
            for(Text txt : values){
                in_doc_cnt++;
                cur_list.add(txt.toString());
            }

            StringBuilder builder = new StringBuilder();
            for(String str : cur_list){
                String[] pair = str.split(dlt);
                String filename = pair[0];
                double word_tf = Double.parseDouble(pair[1]);
                double tf_idf = word_tf*Math.log10((double)corpus_size/in_doc_cnt);
                if(tf_idf != 0){
                    builder.append(cur_id);
                    builder.append(dlt);
                    builder.append(filename);
                    builder.append(dlt);
                    builder.append(tf_idf);
                    myVal.set(builder.toString());
                    context.write(key, myVal);
                    builder.setLength(0);
                }
            }
        }
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();

        //set corpus size
        conf.setLong("mapreduce.task.timeout", 10000*60*60);   
 
        Job job = new Job(conf, "TFIDFCal");
        job.setJarByClass(TFIDFCal.class);
      
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setMapperClass(TFIDFCal.TFIDFCalMap.class);
        job.setReducerClass(TFIDFCal.TFIDFCalReduce.class);

        job.setNumReduceTasks(1);

        for(int i=0;i<args.length-1;i++){
            FileInputFormat.addInputPath(job, new Path(args[i]));
        }
        FileOutputFormat.setOutputPath(job, new Path(args[args.length-1]));
        
        return job.waitForCompletion(true)? 0 : 1;
    }

    public static void main(String[] args) throws Exception{
        int res = ToolRunner.run(new Configuration(), new TFIDFCal(), args);
        System.exit(res);
    }
}
