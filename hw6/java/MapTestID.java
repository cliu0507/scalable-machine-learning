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



public class MapTestID extends Configured implements Tool{
    public static String dlt = " ";

    public static class MapTestIDMap extends Mapper<LongWritable, Text, Text, Text> {
        public static Text myKey = new Text();
        public static Text myVal = new Text();

        public void map(LongWritable key, Text value, Context context) throws IOException, InterruptedException{
            String cur_file = ((FileSplit) context.getInputSplit()).getPath().getParent().getParent().getName();
            String train_file = context.getConfiguration().get("train_file");
            if(cur_file.equals(train_file)){
                StringTokenizer st = new StringTokenizer(value.toString());
                String word = st.nextToken();
                String f_id = st.nextToken();
                myKey.set(word);
                myVal.set(f_id);
                context.write(myKey, myVal);
            }else{
                StringTokenizer st = new StringTokenizer(value.toString());
                String word = st.nextToken();
                String f_id = st.nextToken();
                StringBuilder builder = new StringBuilder(dlt);
                while(st.hasMoreTokens()){
                    String filename = st.nextToken();
                    String tf_idf = st.nextToken();
                    builder.append(filename);
                    builder.append(dlt);
                    builder.append(tf_idf);
                    builder.append("\t");
                }
                myKey.set(word);
                myVal.set(builder.toString());
                context.write(myKey, myVal); 
            }
        }
    }

    public static class MapTestIDReduce extends Reducer<Text, Text, Text, Text> {
        public static Text myVal = new Text();

        public void reduce(Text key, Iterable<Text> values, Context context) throws IOException, InterruptedException{
            
            String[] pair = new String[2];
            int count = 0;
            for(Text txt : values){
                pair[count] = txt.toString();
                count++;
            }

            //word exists in training
            if(count==2){
                StringTokenizer st_one, st_two;
                if(pair[0].contains(dlt)){
                    st_one = new StringTokenizer(pair[1]);
                    st_two = new StringTokenizer(pair[0]);    
                }else{
                    st_one = new StringTokenizer(pair[0]);
                    st_two = new StringTokenizer(pair[1]);
                }

                //outputting the data
                String f_id = st_one.nextToken();
               
                StringBuilder builder = new StringBuilder(dlt);
                builder.append(f_id);
                builder.append(dlt); 
                while(st_two.hasMoreTokens()){
                    String filename = st_two.nextToken();
                    String tf_idf = st_two.nextToken();
                    builder.append(filename);
                    builder.append(dlt);
                    builder.append(tf_idf);
                    builder.append("\t");
                }
                myVal.set(builder.toString());
                context.write(key, myVal);
            }
            
        }
    }

    public int run(String[] args) throws Exception {
        Configuration conf = getConf();
        conf.setLong("mapreduce.task.timeout", 10000*60*60);

        Path train_file = new Path(args[0]);
        Path test_file = new Path(args[1]);
        conf.set("train_file", train_file.getParent().getName());

        Job job = new Job(conf, "MapTestID");
        job.setJarByClass(MapTestID.class);

        
        job.setOutputKeyClass(Text.class);
        job.setOutputValueClass(Text.class);

        job.setMapperClass(MapTestID.MapTestIDMap.class);
        job.setReducerClass(MapTestID.MapTestIDReduce.class);

        job.setNumReduceTasks(300);

        FileInputFormat.addInputPath(job, train_file);
        FileInputFormat.addInputPath(job, test_file);
        FileOutputFormat.setOutputPath(job, new Path(args[2]));

        return job.waitForCompletion(true)? 0 : 1;
    }

    public static void main(String[] args) throws Exception{
        int res = ToolRunner.run(new Configuration(), new MapTestID(), args);
        System.exit(res);
    }
}
