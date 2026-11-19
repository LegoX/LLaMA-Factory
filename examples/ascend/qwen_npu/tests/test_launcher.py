import importlib.util,json,tempfile,unittest
from pathlib import Path
import yaml
spec=importlib.util.spec_from_file_location('launcher',str(Path(__file__).resolve().parents[1] / 'image/bin/launcher.py'));module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class ConfigurationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
  (self.root/'model').mkdir();(self.root/'model/config.json').write_text('{}')
  (self.root/'data').mkdir();(self.root/'data/dataset_info.json').write_text('{}')
  (self.root/'hyper.json').write_text('{}')
  self.config={'model_name_or_path':'model','dataset_dir':'data','output_dir':'output','do_train':True,'use_hyper_parallel':True,'hyper_parallel_args':'hyper.json','hyper_parallel_cp_size':4,'launcher':{'nproc_per_node':16}}
 def prepare(self):
  p=self.root/'job.yaml';p.write_text(yaml.safe_dump(self.config));return module.prepare(p)
 def test_relative_paths_and_card_count(self):
  train,dist,_=self.prepare();self.assertEqual(train['model_name_or_path'],str(self.root/'model'));self.assertEqual(dist['num_processes'],16);self.assertNotIn('launcher',train);self.assertFalse((self.root/'output').exists())
 def test_invalid_cp_rejected(self):
  self.config['launcher']['nproc_per_node']=2
  with self.assertRaisesRegex(ValueError,'divisible'):self.prepare()
 def test_legacy_profile_rejected(self):
  self.config['launcher']['profile']='qwen36_fsdp2'
  with self.assertRaisesRegex(ValueError,'migration'):self.prepare()
 def test_existing_output_protected(self):
  (self.root/'output').mkdir();(self.root/'output/result').write_text('keep')
  with self.assertRaisesRegex(ValueError,'Nonempty'):self.prepare()
 def test_conflicting_topology_rejected(self):
  (self.root/'dist.yaml').write_text(yaml.safe_dump({'num_machines':1,'num_processes':8,'distributed_type':'FSDP'}))
  self.config['launcher']['accelerate_config']='dist.yaml'
  with self.assertRaisesRegex(ValueError,'conflicts'):self.prepare()
 def test_existing_token_cache(self):
  self.config.pop('dataset_dir');(self.root/'tokens').mkdir();(self.root/'tokens/dataset_dict.json').write_text('{}');self.config['tokenized_path']='tokens';self.prepare()
if __name__=='__main__':unittest.main()
