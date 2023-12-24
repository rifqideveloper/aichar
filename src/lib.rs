#![feature(adt_const_params)]
#![allow(dead_code)]

use std::{
	sync::{Arc, RwLock, mpsc::{channel,Sender, Receiver} }
	,thread
	,fs::File
	,fs
	,io::{Read,Write}
	,f64::consts::E
};
use rand::prelude::*;

//<<<<<<<<<<<<<gradian_desent>>>>>>>>>>>>>>>>>
#[macro_export]
macro_rules! linear {
	(activation, $x:expr ,$len:expr) => {
		$x.iter_mut().for_each(|i| 
			*i *= 2f64
		);
	};
	(derivative, $output:expr ) => {
		panic!("linear not ready");
	};
}
#[macro_export]
macro_rules! Leaky_ReLU {
	(activation, $x:expr ,$len:expr) => {
		$x.iter_mut().for_each(|i| 
			*i = if *i >= 0f64{
                *i
            }else{
				*i / 20f64
			}
		);
	};
	(derivative, $output:expr ) => {
		if $output >= 0f64{
			1f64
        } else {
			1f64 / 20f64
		}
	};
}
#[macro_export]
macro_rules! argmax {
	(activation, $x:expr ,$len:expr) => {
		panic!("argmax not ready");
	};
	(derivative, $output:expr ) => {
		panic!("argmax not ready");
	};
}
#[macro_export]
macro_rules! tanh {
	(activation, $x:expr ) => {
		$x.iter_mut().for_each(|i| *i = i.tanh() );
	};
	(derivative, $output:expr ) => {
		{
		let i = $output.tanh() ;
			1.0 - (i*i)
		}
	};
}
#[macro_export]
macro_rules! softmax {
	(activation, $input:expr ,$len:expr) => {
		{
		let mut m : f64 = f64::NEG_INFINITY;
		for i in 0..$len {
			if $input[i] > m {
				 m = $input[i];
			}
		}
		let mut sum : f64 = 0.0f64;
		for i in 0..$len {
			sum += ($input[i] - m).exp();
		}
		let mut offset : f64 = m + E.log({sum});
		for i in 0..$len {
			$input[i] = ($input[i] - offset).exp();
		}
		}
	};
	(derivative , $output:expr ) => {
		let split_y = $output.chunks(2);
		
		panic!("softmax not ready");
	};
	(derivative , $output:expr ,$error:expr) => {
		let dom = -$output[0];
		$output[0] = $output[0] * (1f64 - $output[0]);
		$error += $output[0];
		for i in 1..$output.len() {
			$output[0] = dom * $output[i];
			$error += $output[i];
		}
		//panic!("softmax not ready");
	};
}
#[macro_export]
macro_rules! sigmoy {
	
	(activation, $x:expr ) => {
		1.0 / (1.0 + E.powf(-$x))
	};
	(activation + iter - simd, $x:expr ,$y:expr) => {
		$x.iter_mut().for_each(|i| *i = sigmoy!(activation,*i) );
	};
	(derivative , $output:expr ) => {
		$output * (1.0 - $output)
	};
}
#[derive(PartialEq,std::marker::ConstParamTy, Eq)]
pub enum gradian_desent_type {
	sigmoy,
	softmax,
	tanh,
	argmax,
	leaky_relu
} impl gradian_desent_type  {
	pub fn activation(& self,input:&mut [f64]) {
		match self {
			gradian_desent_type::sigmoy => { sigmoy!(activation + iter - simd, *input ,input.len()); }
			gradian_desent_type::softmax => { softmax!(activation,*input,input.len()); }
			gradian_desent_type::tanh => {tanh!(activation,*input)}
			gradian_desent_type::argmax => {argmax!(activation,*input,input.len())}
			gradian_desent_type::leaky_relu => {Leaky_ReLU!(activation,*input,input.len())}
		}
		
	}
	pub fn derivative<const HR: usize>(& self,input:&mut [f64;HR],sum_error:&mut f64) {//where LaneCount<{ HR }> : SupportedLaneCount  {
		match self {
			gradian_desent_type::sigmoy => { 
				input.iter_mut()
					.for_each(|delta| {*delta = sigmoy!(derivative, *delta ) ;*sum_error += *delta;} );
			}
			gradian_desent_type::softmax => {  
				softmax!(derivative, input, *sum_error);
			}
			gradian_desent_type::tanh => {
				input.iter_mut()
					.for_each(|delta| {*delta = tanh!(derivative, *delta ) ;*sum_error += *delta;} );
			}
			gradian_desent_type::argmax => {panic!()}
			gradian_desent_type::leaky_relu => {
				input.iter_mut()
					.for_each(|delta| {*delta = Leaky_ReLU!(derivative, *delta ) ;*sum_error += *delta;} );
			}
		}
		
	}
	pub fn derivative_2<const HR: usize>(& self,input:&[f64;HR]) -> [f64;HR]  {
		match self {
			gradian_desent_type::sigmoy => { 
				input.iter()
					.map(|x| sigmoy!(derivative,x) )
					.collect::<Vec<_>>().try_into().unwrap()
			}
			gradian_desent_type::softmax => {  
				panic!()
				//softmax!(derivative, input)
			}
			gradian_desent_type::tanh => {
				input.iter()
					.map(|x| tanh!(derivative, x ) )
					.collect::<Vec<_>>().try_into().unwrap()
			}
			gradian_desent_type::argmax => { panic!()}
			gradian_desent_type::leaky_relu => {panic!()}
		}
		
	}
}
//<<<<<<<<<<<<<basic neuron>>>>>>>>>>>>>>>>>>>
pub fn matrix_mul_no_bias<const Input: usize,const Output: usize>(input:&[f64;Input],output:&mut[f64;Output],output_weight:&[[f64;Input];Output]){
	for x in 0..Output {
		output[x] = input[0] * output_weight[x][0];
		for y in 1..Input {
			output[x] += input[y] * output_weight[x][y];
		}
	}
}
pub fn matrix_mul<const Input: usize,const Output: usize>(input:&[f64;Input],output:&mut[f64;Output],output_weight:&[[f64;Input];Output],bias:&[f64;Output]){
	for x in 0..Output {
		output[x] = bias[x] + input[0] * output_weight[x][0];
		for y in 1..Input {
			output[x] += input[y] * output_weight[x][y];
		}
	}
}
pub fn find_bigger_number_in_array(array:&[f64]) -> usize {
	let mut index = 0usize;
	for x in 1..array.len() as usize {
		index = (index * (array[index] < array[x] ) as usize) + (x * (array[index] > array[x]) as usize)
	}
	index
}
pub struct neuron_network<const I: usize,const HR: usize,const HL: usize,const WE: usize,const OU: usize> {
	pub matrix : ([[f64;HR];HL],[f64;OU]),
	need_update:bool
}impl <const I: usize,const HR: usize,const HL: usize,const WE: usize,const OU: usize> neuron_network <{I},{HR},{HL},{WE},{OU}> {
	pub fn update_output_weight_bias(&mut self,delta:[f64;OU],weight:&mut ([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU]),bias:&mut [f64;OU]) {
		self.need_update = true;
		for x in 0..OU {
			bias[x] -= delta[x] ;
			for y in 0..HR{
				weight.2[x][y] -= delta[x]
			}
		}
	}
	pub fn update_output_weight_bias_transfomer(&mut self,weight:&mut ([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU]),bias:&mut [f64;HR]) {
		self.need_update = true;
		for x in 0..HR {
			weight.0[x][0] -= self.matrix.0[0][x];
			bias[x] -= self.matrix.0[0][x] ;
			for y in 1..I {
				weight.0[x][y] -= self.matrix.0[0][x];
			}
		}
	}
	pub fn get_loss<const gradian_desent:gradian_desent_type>(&mut self,expected:[f64;OU]) -> (f64,[f64;OU]){
		let mut sum_error : f64 = 0.0 ;
		for i in 0..OU {
			self.matrix.1[i] -= expected[i];
			sum_error += self.matrix.1[i]
		}
		let mut delta : [f64;OU] =  self.matrix.1 ;
	
		(sum_error,delta)
		
	}
	pub fn get_loss_transfomer<const gradian_desent:gradian_desent_type>(&mut self,expected:[f64;HR]) -> f64{
		for i in 0..HR {
			self.matrix.0[0][i] -= expected[i]
		}
		
		let mut sum_error : f64 = 0.0 ;
		gradian_desent.derivative::<HR>(&mut self.matrix.0[0] ,&mut sum_error);
		sum_error
	}
	pub fn matrix_mul<const Input: usize,const Output: usize>(&mut self,index:usize,output_weight:&[[f64;Input];Output],bias:&[f64;Output]){
		let offset = index - 1;
		for x in 0..Output {
			self.matrix.0[index][x] = bias[x] + self.matrix.0[offset][0] * output_weight[x][0];
			for y in 1..Input {
				self.matrix.0[index][x] += self.matrix.0[offset][y] * output_weight[x][y];
			}
		}
	}
	pub fn run_enbeding_spacial<const gradian_desent:gradian_desent_type>(&mut self,index:usize,weight :&([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU]),bias :&([[f64;HR];HL],[f64;OU])){
		for x in 0..HR {
			self.matrix.0[0][x] = bias.0[0][x] + weight.0[x][index]
		}
	}
	pub fn run_enbeding_spacial_no_bias<const gradian_desent:gradian_desent_type>(&mut self,index:usize,weight :&([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU])){
		for x in 0..HR {
			self.matrix.0[0][x] = weight.0[x][index]
		}
	}
	pub fn run<const gradian_desent:gradian_desent_type>(&mut self,input:&[f64;I],weight :&([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU]),bias :&([[f64;HR];HL],[f64;OU])){
		self.run_mini::<gradian_desent>(input,weight,bias);
		matrix_mul::<{HR},{OU}>(&{self.matrix.0[WE]},&mut self.matrix.1,&weight.2,&bias.1);
			gradian_desent.activation(&mut self.matrix.1);
	}
	pub fn run_mini<const gradian_desent:gradian_desent_type>(&mut self,input:&[f64;I],weight :&([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU]),bias :&([[f64;HR];HL],[f64;OU])){
		matrix_mul::<{I},{HR}>(input,&mut self.matrix.0[0],&weight.0,&bias.0[0]);
			gradian_desent.activation(&mut self.matrix.0[0]);
	}
	fn randomaize_weight(&mut self,n:f64,m:f64,weight:&mut ([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU]) ) {
		let mut rng = rand::thread_rng();
		
		for y in 0..HR {	
			for x in 0..I {
				weight.0[y][x] = rng.gen_range(n..m);
			}
			for x in 0..OU {
				weight.2[x][y] = rng.gen_range(n..m);
			}
			for x in 0..WE {
				for z in 0..HR {
					weight.1[x][y][z] = rng.gen_range(n..m);
				}
			}
		}
		
	}
	pub fn from_file(name:&str) -> (neuron_network <I,HR,HL,WE,OU>,([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU]),([[f64;HR];HL],[f64;OU])){
		let mut v : neuron_network <I,HR,HL,WE,OU> = neuron_network {
			matrix:unsafe { std::mem::MaybeUninit::uninit().assume_init() },
			need_update:false
		};
		let mut weight :([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU]) = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
		let mut bias :([[f64;HR];HL],[f64;OU]) = unsafe { std::mem::MaybeUninit::uninit().assume_init() };
		if let Ok(mut file) = File::open(&format!("{name}.weight.0")) {
			let mut rng = rand::thread_rng();
			let mut buffer = Vec::with_capacity(file.metadata().unwrap().len().try_into().unwrap());
			file.read_to_end(&mut buffer).unwrap();
			if let Ok(data) = bincode::deserialize::<Vec<Vec<f64>> >(&buffer[..]) {
				for x in 0..HR {
					if let Some(data) = data.get(x) {
						for y in 0..I {
							weight.0[x][y] = if let Some(data) = data.get(y) {
								 *data
							} else {
								v.need_update = true;
								rng.gen_range(0f64..1f64)
							}
						}
						continue
					}
					v.need_update = true;
					for y in 0..I {
						weight.0[x][y] = rng.gen_range(0f64..1f64)
					}
				}
			} else {
				panic!()
			} 
			buffer.clear();
			if let Ok(mut file) = File::open(&format!("{name}.weight.1")) {
				file.read_to_end(&mut buffer).unwrap();
				if let Ok(data) = bincode::deserialize::<Vec<Vec<Vec<f64>>> >(&buffer[..]) {
					for x in 0..WE {
						for y in 0..HR {
							for z in 0..HR {
								if let Some(data) = data.get(x) {
									if let Some(data) = data.get(y) {
										if let Some(data) = data.get(z) {
											
											weight.1[x][y][z] = *data;
											continue
										}
									}
								}
								v.need_update = true;
								weight.1[x][y][z] = rng.gen_range(0f64..1f64)
							}
						}
					}
				} else {
					panic!()
				} 
			} else {
				panic!()
			} 
			buffer.clear();
			if let Ok(mut file) = File::open(&format!("{name}.weight.2")) {
				file.read_to_end(&mut buffer).unwrap();
				if let Ok(data) = bincode::deserialize::<Vec<Vec<f64>> >(&buffer[..]) {
					for x in 0..OU {
						if let Some(data) = data.get(x) {
							for y in 0..HR {
								weight.2[x][y] = if let Some(data) = data.get(y) {
									 *data
								} else {
									v.need_update = true;
									rng.gen_range(0f64..1f64)
								}
							}
							continue
						}
						v.need_update = true;
						for y in 0..HR {
							weight.2[x][y] = rng.gen_range(0f64..1f64)
						}
					}
				} else {
					panic!()
				} 
			} else {
				panic!()
			} 
		} else {
			v.need_update = true;
			v.randomaize_weight(0f64,1f64,&mut weight);
		}
		if let Ok(mut file) = File::open(&format!("{name}.bias.0")) {
			let mut buffer = Vec::with_capacity(file.metadata().unwrap().len().try_into().unwrap());
			file.read_to_end(&mut buffer).unwrap();
			if let Ok(data) = bincode::deserialize::<Vec<Vec<f64>> >(&{buffer}[..]) {
				for x in 0..HL {
					if let Some(data) = data.get(x) {
						for y in 0..HR {
							bias.0[x][y] = if let Some(data) = data.get(y) {
								 *data
							} else {
								v.need_update = true;
								0f64
							}
						}
						continue
					}
					v.need_update = true;
					bias.0[x] = [0f64;HR]
				}
			} else {
				panic!()
			} 
		} else {
			v.need_update = true;
			bias.0 = [[0f64;HR];HL]
		}
		if let Ok(mut file) = File::open(&format!("{name}.bias.1")) {
			let mut buffer = Vec::with_capacity(file.metadata().unwrap().len().try_into().unwrap());
			file.read_to_end(&mut buffer).unwrap();
			if let Ok(data) = bincode::deserialize::<Vec<f64> >(&{buffer}[..]) {
				for x in 0..OU {
					bias.1[x] = if let Some(data) = data.get(x) {
						*data
					} else {
						v.need_update = true;
						0f64
					}
				}
			} else {
				panic!()
			}
		} else {
			v.need_update = true;
			bias.1 = [0f64;OU]
		}
		
		
		(v,weight,bias)
	}
	pub fn save(&self,name:&str,weight:&([[f64;I];HR],[[[f64;HR];HR];WE],[[f64;HR];OU]),bias:&([[f64;HR];HL],[f64;OU])) {
		if !self.need_update{return}
		if let Ok(mut file) = File::create(&format!("{name}.weight.0")) {
			let encoded: Vec<u8> = bincode::serialize(&weight.0.iter().map(|i| Vec::from(*i)).collect::<Vec<Vec<f64>>>()).unwrap();
				file.write_all(&encoded).unwrap()
		}
		if let Ok(mut file) = File::create(&format!("{name}.weight.1")) {
			let encoded: Vec<u8> = bincode::serialize(&weight.1.iter().map(|i| i.iter().map(|i| Vec::from(*i) ).collect() ).collect::<Vec<Vec<Vec<f64>>>>()).unwrap();
				file.write_all(&encoded).unwrap()
		}
		if let Ok(mut file) = File::create(&format!("{name}.weight.2")) {
			let encoded: Vec<u8> = bincode::serialize(&weight.2.iter().map(|i| Vec::from(*i) ).collect::<Vec<Vec<f64>>>()).unwrap();
				file.write_all(&encoded).unwrap()
		}
		if let Ok(mut file) = File::create(&format!("{name}.bias.0")) {
			let encoded: Vec<u8> = bincode::serialize(&bias.0.iter().map(|i| Vec::from(*i)).collect::<Vec<Vec<f64>>>()).unwrap();
				file.write_all(&encoded).unwrap()
		}
		if let Ok(mut file) = File::create(&format!("{name}.bias.1")) {
			let encoded: Vec<u8> = bincode::serialize(&Vec::from(bias.1)).unwrap();
				file.write_all(&encoded).unwrap()
		}
	}
}

//<<<<<<<<<<<<<basic transfomer>>>>>>>>>>>>>>>>>>>
pub struct QueryValueKey <const HR: usize,const MAX_OUTPUT:usize>{
	buffer:Vec<[[f64;HR];2]>,
	pub query_weight:[[f64;HR];HR],
	pub key_weight:[[f64;HR];HR],
	pub value_weight:[[f64;HR];HR],
	co_sin_wave : [f64;HR],
	co_sin_wave_len : usize,
	need_update : bool
} impl <const HR: usize,const MAX_OUTPUT:usize> QueryValueKey <{HR},{MAX_OUTPUT}> {
	pub fn clear(&mut self) {
		//println!("test");
		self.buffer.clear();
		self.co_sin_wave_len = 0
	}
	pub fn run(&mut self,value:&[f64;HR],start:bool,resedual_connection:&mut [f64;HR]) {
		let mut v = ([0f64;HR],[0f64;HR],[0f64;HR]);
		for x in 0..HR {
			(0..HR).for_each(|y| {
				v.0[y] += value[y] * self.query_weight[x][y];
				v.1[y] += value[y] * self.value_weight[x][y] ;
				v.2[y] += value[y] * self.key_weight[x][y]
			})
		}
		if start {
			let mut self_attention = Vec::with_capacity(self.buffer.len()+1);
			let mut sum_num = v.2[0] * v.0[0];
			
			for x in 1..HR {
				sum_num += v.2[x] * v.0[x];
			}
			self_attention.push(sum_num);
			for buffer in self.buffer.iter() {
				sum_num = buffer[1][0] * v.0[0];
				for x in 1..HR {
					sum_num += buffer[1][x] * v.0[x];
				}
				self_attention.push(sum_num);
			}
			gradian_desent_type::softmax.activation(&mut self_attention);
			for x in 0..HR {
				resedual_connection[x] += v.1[x] * self_attention[0]
			}
			for x in 1.. self.buffer.len() {
				for y in 0..HR {
					resedual_connection[y] += self.buffer[x][0][y] * self_attention[x]
				}
			}
			//println!("{self_attention:?}")
		}
		self.buffer.push([v.1,v.2]);
		
	}
	pub fn run_positional_encoding(&mut self,value:&mut [f64;HR]) -> [f64;HR] {
		self.positional_encoding_matrix(HR,100.);
		value.iter_mut().zip(self.co_sin_wave.iter()).for_each(|(x,y)| *x += y);
		*value
	}
	pub fn positional_encoding_matrix(&mut self,d:usize, n:f64) {
		self.co_sin_wave = [0f64;HR];
		let dominator = self.co_sin_wave_len as f64;
		(0..d/2).for_each(|i| {
			let dominator : f64 = dominator / n.powf((2*i/d)as f64) ;
			self.co_sin_wave[2*i] = dominator.sin();
			self.co_sin_wave[2*i+1] = dominator.cos();
		});
		self.co_sin_wave_len += 1
	}
	pub fn init(name:&str,n:f64,m:f64) -> QueryValueKey <HR,MAX_OUTPUT> {
		let mut rng = rand::thread_rng();
		let mut qvk = QueryValueKey{
			buffer:Vec::with_capacity(MAX_OUTPUT)
			,query_weight:[[0f64;HR];HR]
			,key_weight:[[0f64;HR];HR]
			,value_weight:[[0f64;HR];HR]
			,co_sin_wave : [0f64;HR]
			,co_sin_wave_len: 0,
			need_update: false
		};
		let query :Vec<Vec<f64>>= if let Ok(mut data) = File::open(&format!("{name}.query_weight")) {
			let mut buffer = Vec::new();
			data.read_to_end(&mut buffer).unwrap();
			if let Ok(data) = bincode::deserialize::<Vec<Vec<f64> > >(&buffer[..]) {
				data
			} else {
				qvk.need_update = true;
				Vec::new()
			}
		} else{
			qvk.need_update = true;
			Vec::new()
		};
		for x in 0..HR {
			if query.get(x).is_some() {
				for y in 0..HR {
					qvk.query_weight[x][y] = if let Some(z) = query[x].get(y) {
						*z
					} else {
						qvk.need_update = true;
						rng.gen_range(n..m)
					}
				}
			} else {
				qvk.need_update = true;
				qvk.query_weight[x].iter_mut().for_each(|y| *y = rng.gen_range(n..m) )
			}
		}
		drop(query);
		let value :Vec<Vec<f64>>= if let Ok(mut data) = File::open(&format!("{name}.value_weight")) {
			let mut buffer = Vec::new();
			data.read_to_end(&mut buffer).unwrap();
			if let Ok(data) = bincode::deserialize::<Vec<Vec<f64> > >(&buffer[..]) {
				data
			} else {
				qvk.need_update = true;
				Vec::new()
			}
		} else{
			qvk.need_update = true;
			Vec::new()
		};
		for x in 0..HR {
			if value.get(x).is_some() {
				for y in 0..HR {
					qvk.value_weight[x][y] = if let Some(z) = value[x].get(y) {
						*z
					} else {
						qvk.need_update = true;
						rng.gen_range(n..m)
					}
				}
			} else {
				qvk.need_update = true;
				qvk.value_weight[x].iter_mut().for_each(|y| *y = rng.gen_range(n..m) )
			}
		}
		drop(value);
		
		let key :Vec<Vec<f64>>= if let Ok(mut data) = File::open(&format!("{name}.key_weight")) {
			let mut buffer = Vec::new();
			data.read_to_end(&mut buffer).unwrap();
			if let Ok(data) = bincode::deserialize::<Vec<Vec<f64> > >(&buffer[..]) {
				data
			} else {
				qvk.need_update = true;
				Vec::new()
			}
		} else{
			qvk.need_update = true;
			Vec::new()
		};
		for x in 0..HR {
			if key.get(x).is_some() {
				for y in 0..HR {
					qvk.key_weight[x][y] = if let Some(z) = key[x].get(y) {
						*z
					} else {
						qvk.need_update = true;
						rng.gen_range(n..m)
					}
				}
			} else {
				qvk.need_update = true;
				qvk.key_weight[x].iter_mut().for_each(|y| *y = rng.gen_range(n..m) )
			}
		}
		drop(key);
		
		qvk
	}
	pub fn save(&self,name:&str) {
		if !self.need_update {return}
		if let Ok(mut file) = File::create(&format!("{name}.query_weight")) {
			let mut v = Vec::with_capacity(HR);
			for x in 0..HR {
				v.push( Vec::from(self.query_weight[x]) );
			}
			let encoded: Vec<u8> = bincode::serialize(&{v}).unwrap();
				file.write_all(&encoded).unwrap()
		}
		if let Ok(mut file) = File::create(&format!("{name}.value_weight")) {
			let mut v = Vec::with_capacity(HR);
			for x in 0..HR{
				v.push( Vec::from(self.value_weight[x]) );
			}
			let encoded: Vec<u8> = bincode::serialize(&{v}).unwrap();
				file.write_all(&encoded).unwrap()
		}
		if let Ok(mut file) = File::create(&format!("{name}.key_weight")) {
			let mut v = Vec::with_capacity(HR);
			for x in 0..HR{
				v.push( Vec::from(self.key_weight[x]) );
			}
			let encoded: Vec<u8> = bincode::serialize(&{v}).unwrap();
				file.write_all(&encoded).unwrap()
		}
	}
}
pub struct Transformer_enbed<const I: usize,const HR: usize> {
	weight:[[[f64;HR];HR];I],
	need_update : bool
}impl <const I: usize,const HR: usize> Transformer_enbed <{I},{HR}> {
	fn get<const gradian_desent:gradian_desent_type>(&self,index:usize) -> [f64;HR] {
		let mut v = self.weight[index][0];
		for x in 1..HR {
			for y in 0..HR {
				v[y] += self.weight[index][x][y]
			}
		}
		gradian_desent.activation(&mut v);
		v
	}
	fn init(name:&str)-> Transformer_enbed <I,HR>{
		let mut v = Transformer_enbed{weight:[[[0f64;HR];HR];I],need_update:false};
		let mut rng = rand::thread_rng();
		if let Ok(mut data) = File::open(name) {
			let mut buffer = Vec::new();
			data.read_to_end(&mut buffer).unwrap();
			if let Ok(data) = bincode::deserialize::<Vec<Vec<Vec<f64>> > >(&buffer[..]) {
				for x in 0..I {
					for y in 0..HR {
						for z in 0..HR {
							if let Some(data) = data.get(x) {
								if let Some(data) = data.get(y) {
									if let Some(data) = data.get(z) {
										v.weight[x][y][z] = *data;
										continue
									}
								}
							}
							v.need_update = true;
							v.weight[x][y][z] = rng.gen_range(0f64..1f64)
						}
					}
				}
			} else {
				v.need_update = true;
			}
		} else{
			v.need_update = true;
		};
		
		
		v
	}
	pub fn save(&self,name:&str){
		if !self.need_update {return}
		if let Ok(mut file) = File::create(name) {
			let mut v = Vec::with_capacity(HR);
			for i in 0..I{
				let mut vv = Vec::with_capacity(HR);
				for x in 0..HR {
					vv.push(Vec::from(self.weight[i][x]));
				}
				v.push(vv);
			}
			let encoded: Vec<u8> = bincode::serialize(&{v}).unwrap();
				file.write_all(&encoded).unwrap()
		}
	}
}
pub struct Transformer<const I: usize,const HR: usize,const HL: usize,const WE: usize,const MAX_OUTPUT: usize> {
	input:Sender<(Option<usize>,Option<(usize,bool)>)>,
	output:Receiver<usize>,
	thread_join_neuron:Option < std::thread::JoinHandle<()> >,
	thread_join_query_value_key:Option <std::thread::JoinHandle<()>>,
	thread_join_transformer_enbeding:Option < std::thread::JoinHandle<()>>
}impl <const I: usize,const HR: usize,const HL: usize,const WE: usize ,const MAX_OUTPUT: usize>Transformer <{I},{HR},{HL},{WE},{MAX_OUTPUT}> {
	pub fn training(&mut self ,input_:&[String],str_arr:&[&str],max_:usize) -> usize {
		let mut data : Vec<usize> = Vec::new();
		for i in 0..input_.len() {
			for i in input_[i].split_whitespace() {
				for x in 1..str_arr.len() {
					if i.starts_with(str_arr[x]) {
						data.push(x);
						break
					}
				}
			}
			data.push(0);
		}
		let stop = data.len() - 1;
		
		for _ in 0..max_ {
			for i in 1..data.len() {
				self.input.send(
					(Some(data[i-1]),if let Some(data) = data.get(i) {Some((*data,stop == i))} else {panic!()}) 
				).unwrap();
			}
		}
		/*
		for i in 1..data.len()  {
			if  let Ok(data) = self.output.recv() {
			//print!("{} ",str_arr[data]);	
			}
		}
		*/
		//println!("test{data:?}");
		data.len() - 1
	}
	pub fn print(&mut self ,mut input_: String,str_arr:&[&str]){
		//print!("test<2>");
		for i in input_.split_whitespace() {
			for x in 1..str_arr.len() {
				if i.starts_with(str_arr[x]){
					self.input.send((Some(x),None)).unwrap();
					break
				}
			}
		}
		self.input.send((Some(0),None)).unwrap();
		
		while let Ok(data) = self.output.recv() {
			print!("{} ",str_arr[data]);
			if data == 0 { break }
		}
		
	}
	pub fn init(f_name:&'static str) -> Transformer<I, HR, HL, WE,MAX_OUTPUT>{
		let (input_enbed,input_enbed_input) = channel::<(Option<usize>,Option<(usize,bool)>)>();
		let (output,output_output) = channel::<usize>();
		let (input_to_QueryValueKey,input_to_QueryValueKey_output) = channel::<([f64;HR],Option<(usize,bool)>,bool)>();
		let (input_to_neuron,input_to_neuron_output) = channel::<([f64;HR],Option<usize>)>();
		let (neuron_to_enbed,neuron_to_enbed_output) = channel::<[f64;I]>();
		let (backprop,backprop_output) = channel::<f64>();
		let (backprop2,backprop_output2) = channel::<f64>();
		Transformer {
			input:input_enbed
			,output:output_output
			,thread_join_transformer_enbeding : Some( thread::Builder::new()
				.name("QueryValueKey node".into())
				.stack_size(100 * 1024 * 1024)
				.spawn( move || {
					let mut query_value_key = QueryValueKey::<HR,MAX_OUTPUT>::init(format!("{f_name}qvk.bincode").as_str(),0f64,1f64);
					while let Ok((mut data,target,start)) = input_to_QueryValueKey_output.recv() {
						let mut resedual_connection = query_value_key.run_positional_encoding(&mut data);
						query_value_key.run(&data,start,&mut resedual_connection);
						
						data.iter_mut().zip(resedual_connection.iter()).for_each(|(x,y)| *x += y);
						if let Some((target,need_clear)) = target {
							input_to_neuron.send((data,Some(target)));
							if need_clear {
								query_value_key.clear()
							}
							let error = backprop_output.recv().unwrap();
							continue
						}
						if start  {
							input_to_neuron.send((data,None));
						}
						//print!("test<qvk>");
					} 
					query_value_key.save(format!("{f_name}qvk.bincode").as_str());
				}).unwrap())
			,thread_join_query_value_key : Some( thread::Builder::new()
				.name("enbeding node".into())
				.stack_size(100 * 1024 * 1024)
				.spawn( move || {
					let mut enbed = Transformer_enbed::<I,HR>::init(format!("{f_name}enbeding.bincode").as_str());
					loop {
						let mut start = false;
						while !start {
							if let Ok((Some(input) ,target)) = input_enbed_input.recv() {
								//print!("test<enbed>");
								start = input == 0 && !target.is_some();
								let input = enbed.get::<{gradian_desent_type::leaky_relu}>(input);
								input_to_QueryValueKey.send((input,target,start));
								//println!("{input:?}");
								
								if target.is_some() {
									let test = neuron_to_enbed_output.recv().unwrap();
									let mut index = 0;
									for i in 1..test.len() {
										if test[index] < test[i] {
											index = i
										}
									}
									output.send(index);
									let error = backprop_output2.recv().unwrap();
									
									//print!("<{error}> ");
								}
							} else {
								enbed.save(format!("{f_name}enbeding.bincode").as_str());
								return
							}
						}
						for _ in 0..MAX_OUTPUT {
							let test = neuron_to_enbed_output.recv().unwrap();
							let mut index = 0;
							for i in 1..test.len() {
								if test[index] < test[i] {
									index = i
								}
							}
							output.send(index);
							
							if index == 0 {
								//output.send(0);
								break
							} else {
								let index = enbed.get::<{gradian_desent_type::leaky_relu}>(index);
								input_to_QueryValueKey.send((index,None,start));
							}
						}
						output.send(0);
					}
					
				}).unwrap())
			,thread_join_neuron : Some( thread::Builder::new()
				.name("neuron node array".into())
				.stack_size(100 * 1024 * 1024)
				.spawn( move || {
					const gradian : gradian_desent_type = gradian_desent_type::softmax;
					let (mut n,mut weight,mut bias) = neuron_network::<HR,I,1,0,0>::from_file(format!("{f_name}neuron1").as_str());//neuron::<HR,I,1,0,0>::init();
					//n.load("neuron1",0f64,1f64);
					while let Ok((data,target)) = input_to_neuron_output.recv() {
						//print!("test");
						n.run_mini::<gradian>(&data,&weight,&bias);
						
						//println!("{:?}",n.bias);
						
						let _ = neuron_to_enbed.send(n.matrix.0[0]);
						//print!("test");
						if let Some(target) = target {
							let test = find_bigger_number_in_array(&n.matrix.0[0]);
							let mut target2 = [0f64;I];
							target2[target] = 1f64;
							
							let sum_error = n.get_loss_transfomer::<gradian>(target2);
							backprop.send(sum_error).unwrap();
							backprop2.send(sum_error).unwrap();
							if test == target {
								//print!("<ok>");
								continue
							}
							n.update_output_weight_bias_transfomer(&mut weight,&mut bias.0[0]);
						}
					} 
					n.save(format!("{f_name}neuron1").as_str(),&weight,&bias);
				}).unwrap())
		}
	}
}impl <const I: usize,const HR: usize,const HL: usize,const WE: usize,const MAX_OUTPUT:usize> Drop for Transformer <{I},{HR},{HL},{WE},{MAX_OUTPUT}>{
	fn drop(&mut self){
		let _ = self.input.send((None,None));
		self.thread_join_neuron.take().unwrap().join().unwrap();
		self.thread_join_query_value_key.take().unwrap().join().unwrap();
		self.thread_join_transformer_enbeding.take().unwrap().join().unwrap();
	}
}

