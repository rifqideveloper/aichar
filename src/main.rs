#![allow(non_snake_case)]
#![feature(portable_simd)]
#![feature(let_chains)]
#![allow(dead_code)]
#![allow(unused_variables)]

use dioxus::prelude::*;
use std::default::Default;
use std::sync::Arc;
use std::f32::consts::E;
use std::simd::{Simd,LaneCount,SupportedLaneCount};
use rand::prelude::{/*thread_rng,*/Rng};
use serde::{/*Serialize, Deserialize ,*/de};
use std::fs::File;
use std::io::prelude::*;
use std::fs::read_to_string;

macro_rules! sigmoy {
	(activation, $x:expr ) => {
		1.0 / (1.0 + E.powf(-$x))
	};
	(output = activation + simd , $x:expr ) => {
		$x.iter_mut().for_each(|i| *i = E.powf(-*i) );
		let sim = Simd::from([1.0f32;OU]);
		$x = ((Simd::from($x) + sim ) / sim).to_array();
	};
	(derivative , $output:expr ) => {
		$output * (1.0 - $output)
	};
}
/*
	help: the following other types implement trait `SupportedLaneCount`:
              LaneCount<16>
              LaneCount<1>
              LaneCount<2>
              LaneCount<32>
              LaneCount<4>
              LaneCount<64>
              LaneCount<8>
*/
const I : usize = 4;
const HR : usize = 2;
const HL : usize = 2;
const WE : usize = 1;
const OU : usize = 4;

#[derive(Debug,PartialEq,Clone)]
struct Neuron {
	
	matrix : ([f32;I],[[f32;HR];HL],[f32;OU]),
	
	bias :([[f32;HR];HL],[f32;OU]),
	
	weight :([[f32;I];HR],[[[f32;HR];HR];WE]),
	
	delta : ([[f32;HR];HL],[f32;OU]),
	
	token : Vec<(Box<str>,f32,f32,f32)>,
	
	save_name : Box<str>
}
impl Neuron {
	
	pub fn comment<'a>(&'a mut self,c:Arc<[usize]>) -> (Arc<[usize]>,Vec<&'a (String,f32)>){
		let com : Vec<&'a (String,f32)> = Vec::new();
		
		(c,com)
	}
	pub fn token_iq(& mut self,Vec_id:Arc<[usize]>) {
		let m = self.token[Vec_id[0]].1 ;
		for i in 1..Vec_id.len() {
			self.token[Vec_id[i]].1 = m;
		}
	}
	pub fn indexs_to_str(& self,c:Arc<[usize]>) -> String {
		let mut v = String::new(); 
		c.iter().for_each(|i|
			v.push_str(&format!("{} ",self.token[*i].0))
		);
		v
	}
	
	pub fn train (&mut self,input_out:Arc<[([f32;I],[f32;OU])]>,max_loop:usize) -> Vec<f32> {
		let mut total_error = Vec::new();
		for i in input_out.iter() {
			self.matrix.0 = i.0;	
			for _ in 0..max_loop {
				self.run();
				println!("output:{:?}",self.matrix.2);
				let LEARNING_RATE = self.learning(&i.1) ;
				
				if LEARNING_RATE <= 0.01 {
					break
				}
				self.update_weight(LEARNING_RATE);
				
				total_error.push( LEARNING_RATE );
			}
		}
		total_error
	}
	
	pub fn learning(&mut self ,expected:&[f32;OU]) -> f32 {//for_each
		let mut sum_error = 0.0;
		self.delta.1 = (Simd::from(self.matrix.2) - Simd::from(*expected) ).to_array() ;
		self.delta.1.iter_mut()
			.for_each(|delta| {*delta = sigmoy!(derivative, *delta ) ;sum_error += *delta;} );
			
		self.delta.0 = [[0.0;HR];HL];
		
		for j in 0..HR {
			for weight in self.weight.1[WE-1][j] {
				for delta in 0..OU {
					self.delta.0[HL-1][j] -= weight * self.delta.1[delta]
				}
			} 
		}
		let derivative : [f32;HR] = self.matrix.1[HL-1].iter()
			.map(|x| sigmoy!(derivative,x) )
			.collect::<Vec<_>>().try_into().unwrap();
		self.delta.0[HL-1] = ( Simd::from(derivative) * Simd::from(self.delta.0[HL-1]) ).to_array();
		
		for i in (1..HL-1).rev() {
			for j in 0..HR {
				for weight in self.weight.1[i-1][j] {
					for delta in 0..HR {
						self.delta.0[i][j] -= weight * self.delta.0[i+1][delta]
					}
				} 
			}
			let derivative : [f32;HR] = self.matrix.1[i].iter()
				.map(|x| sigmoy!(derivative,x) )
				.collect::<Vec<_>>().try_into().unwrap();
			self.delta.0[i] = ( Simd::from(derivative) * Simd::from(self.delta.0[i]) ).to_array()
		}
		for j in 0..HR {
			for weight in self.weight.0[j] {
				for delta in 0..HR {
					self.delta.0[0][j] -= weight * self.delta.0[1][delta]
				}
			} 
		}
		let derivative : [f32;HR] = self.matrix.1[0].iter()
			.map(|x| sigmoy!(derivative,x) )
			.collect::<Vec<_>>().try_into().unwrap();
			
			
		self.delta.0[0] = ( Simd::from(derivative) * Simd::from(self.delta.0[0]) ).to_array();
		
		sum_error 
	}
	pub fn update_weight(&mut self ,LEARNING_RATE : f32) {
		for i in 0..HR {
			for x in 0..I {
				self.weight.0[i][x] += LEARNING_RATE * self.delta.0[0][i] * self.matrix.0[x];
			}
		}
		let offset = self.matrix.1.len() - 1;
		for offset in 1..offset {
			for i in 0..HR {	
				for x in 0..HR {
					self.weight.1[offset-1][i][x] += LEARNING_RATE * self.delta.0[offset][i] * self.matrix.1[offset-1][i]
				}
			}
		}
		for i in 0..HR {	
			for x in 0..HR {
				self.weight.1[offset-1][i][x] += LEARNING_RATE * self.delta.0[offset][i] * self.matrix.1[offset-1][i]
			}
		}
		
	}
	pub fn strsplit(&mut self,s:&str) -> Vec<String> {
		let (mut v,mut offset) = (Vec::from(["".to_string()]),0);
		s.chars().for_each( |i|
		   match i {
				' ' | '\n' => if &v[offset] != "" {
					v.push("".to_string());
					offset += 1;
				},
			   '.' | ',' | '-' | '–' | '\'' | '(' |')' => 
					offset += if &v[offset] == "" {
						v[offset].push(i);
						
						v.push("".to_string());
						1
					} else {
						v.push(i.to_string());
						v.push("".to_string());
						2
					},
			   _ =>{
				   v[offset].push(i);
			   }
		   }
		);
		if &v[offset] == "" { 
			v.pop() ;
		} 
		v
	}
	
	pub fn str_parse(&mut self,text:&str) -> Arc<[usize]> {
		self.strsplit(text/*.to_lowercase()*/).iter().map(|new_token| 
			if let Some(i) = self.token.iter().rposition(|old_token| old_token.0 == (*new_token).clone().into() ) {
				i
			} else {
				let mut rng = rand::thread_rng();
				let v = self.token.len();
				self.token.push(((*new_token).clone().into(),rng.gen(),rng.gen(),rng.gen()));
				v
			}
		).collect()
	}
	pub fn str_parse_n_comment(&mut self,text:&str) -> (String,Arc<[usize]>) {
		let mut comment = String::new();
		
		let v : Arc<[usize]> = self.strsplit(text).iter().map(|new_token| 
			if let Some(i) = self.token.iter().rposition(|old_token| old_token.0 == (*new_token).clone().into() ) {
				comment.push_str(&format!("{} ",self.token[i].0));
				i
			} else {
				let mut rng = rand::thread_rng();
				let v = self.token.len();
				comment.push_str(&format!("{} ",new_token));
				self.token.push(((*new_token).clone().into(),rng.gen(),rng.gen(),rng.gen()));
				v
			}
		).collect();
		
		(comment,v)
	}
	pub fn token_id(&mut self,text:&str) -> Arc<[(f32,f32,f32)]> {
		self.strsplit(text/*.to_lowercase()*/).iter().map(|new_token| 
			if let Some(i) = self.token.iter().rposition(|old_token| old_token.0 == (*new_token).clone().into() ) {
				(self.token[i].1,self.token[i].2,self.token[i].3)
			} else {
				let mut rng = rand::thread_rng();
				let v = self.token.len();
				let id1 = rng.gen();
				let id2 = rng.gen();
				let id3 = rng.gen();
				self.token.push(((*new_token).clone().into(),id1,id2,id3));
				(id1,id2,id3)
			}
		).collect()
	}
	pub fn koordinat_to_str(koor:&[f32,f32,f32,f32]),buf:&mut String)  {
		
	}
	pub fn dialog_str(&mut self,text:Arc<[usize]>) -> String{
		let mut dialog = "".to_string();
		
		let mode : f32 = 0.0;
		
		text.iter().map(|index|{
			&self.token[indexs]
		}).map(|koor|{
			self.matrix.0[0] = mode;
			self.matrix.0[1] = koor.1;
			self.matrix.0[2] = koor.2;
			self.matrix.0[3] = koor.3;
			self.run();
			&self.matrix.2
		}).for_each(|koordinat|{
			self.koordinat_to_str(koordinat,&mut dialog)
		});
		
		
		dialog
	}
	pub fn reset_token(&mut self) {
		let old = self.token.iter().map(|i| i.0.clone().into() ).collect::<Vec<String>>();
		self.token.clear();
		old.iter().for_each(|i| { self.str_parse(i); } );
	}
	pub fn str_randomize(&mut self) {
		let mut rng = rand::thread_rng();
		self.token.iter_mut().for_each(|token|
			token.1 = rng.gen()
		)
	}
	pub fn download_words( &mut self ,http:&str) -> Result<(),std::boxed::Box<reqwest::Error>>{
		let resp = reqwest::blocking::get(http)?.text()?;
		let mut resp = resp.split_whitespace();
		self.token.clear();
		if let Some(t) = resp.next() {
			self.token.reserve_exact(t.parse::<usize>().unwrap())
		} else {
			panic!()
		}
		loop {
			if let Some(token) = resp.next() {
				if let Some(id1) = resp.next() {
					if let Some(id2) = resp.next() {
						if let Some(id3) = resp.next() {
							self.token.push((
								token.into(),
								id1.parse::<f32>().unwrap(),
								id2.parse::<f32>().unwrap(),
								id3.parse::<f32>().unwrap())
							);
						} else {
							panic!()
						}
					} else {
						panic!()
					}
					
				} else {
					panic!()
				}
			} else {
				break
			}
		}
		Ok(())
	}
	pub fn download_pages ( &mut self ,http:&str) -> Result<Arc<[usize]>,std::boxed::Box<reqwest::Error>>{
		Ok(
			self.str_parse(
				&reqwest::blocking::get(http)?.text()?
			)
		)
	}
	pub fn load_from__web( &mut self ,http:&str)  {
		
	}
	pub fn saving<'a>( &'a self ) -> (
		&'a ([[f32;HR];HL],[f32;OU]),
		&'a ([[f32;I];HR],[[[f32;HR];HR];WE]),
		&'a Vec<(Box<str>,f32,f32,f32)>,
		
	) { 
		(&self.bias,&self.weight,&self.token)
	}
	pub fn save_files_bias(& self,file_name:&str) -> Result< File, std::boxed::Box<std::io::Error> > {
		Ok( File::create(format!("{}.bias.{}.{}.{}",file_name,HR,HL,OU).as_str())? )
	}
	pub fn save_files_weight(& self,file_name:&str) -> Result< File, std::boxed::Box<std::io::Error> > {
		Ok( File::create(format!("{}.weight.{}.{}.{}",file_name,I,HR,WE).as_str())? )
	}
	pub fn save_files_token(& self,file_name:&str)-> Result< File, std::boxed::Box<std::io::Error> > {
		Ok( File::create(format!("{}.token",file_name).as_str())? )
		//file_token.write_all(&token_encoder).unwrap();
	}
	pub fn save_unique_words(& self,file_name:&str) -> Result<(),std::io::Error> {
		let mut file = File::create(format!("{}.uniquetoken",file_name).as_str())?;
		
		file.write_all(format!("{}\n",self.token.len()).as_bytes())?;
		self.token.iter().for_each(|t|{
			file.write_all(format!("{} {} {} {}\n",t.0,t.1,t.2,t.3).as_bytes()).unwrap();
		});
		Ok(())
	}
	pub fn save_sentence(& self,file_name:&str,sentence:Vec<String>) -> Result<(),std::io::Error> {
		let mut file = File::create(format!("{}.sentence",file_name).as_str())?;
		
		sentence.iter().for_each(|t|
			for i in 0..self.token.len() {
				if *self.token[i].0 == *t {
					file.write_all(format!("{}\n", i ).as_bytes()).unwrap();
					break;
				}
			}
		);
		Ok(())
	}
	
	pub fn load_unique_words_from_file (&mut self,name:&str) -> Result<(),std::io::Error> {
		let token = read_to_string(format!("{}.uniquetoken",name).as_str())?;
		let mut token = token.split_whitespace();
		self.token.clear();
		if let Some(token) = token.next() {
			self.token.reserve_exact(token.parse::<usize>().unwrap())
		} else {
			panic!()
		}
		
		loop {
			if let Some(str_token) = token.next() { 
				if let Some(koor1) = token.next() { 
					if let Some(koor2) = token.next() { 
						if let Some(koor3) = token.next() { 
							self.token.push(
								( str_token.into(),
								koor1.parse::<f32>().unwrap(),
								koor2.parse::<f32>().unwrap(),
								koor3.parse::<f32>().unwrap()
								)
							)
						} else {
							panic!()
						}
					} else {
						panic!()
					}
				} else {
					panic!()
				}
			} else {
				break
			};
			
		}
		Ok(())
	}
	
	pub fn load_sentence_from_file (& self,name:&str) -> Vec<usize>{
		let mut result = Vec::new();
		for line in read_to_string(format!("{}.sentence",name).as_str()).unwrap().lines() {
			result.push( line.parse::<usize>().unwrap() );
		}
		result
	}
	
	pub fn load_bincode_bias<T: de::DeserializeOwned>(& self ,name:&str) -> Result<T, std::boxed::Box<bincode::ErrorKind>> {
		let f = std::fs::read(format!("{}.bias.{}.{}.{}",name,HR,HL,OU).as_str())?;
		bincode::deserialize( &f )
	}
	pub fn load_bincode_weight<T: de::DeserializeOwned>(& self ,name:&str) -> Result<T, std::boxed::Box<bincode::ErrorKind>> {
		let f = std::fs::read(format!("{}.weight.{}.{}.{}",name,I,HR,WE).as_str())?;
		bincode::deserialize( &f )
	}
	pub fn load_bincode_token<T: de::DeserializeOwned>(& self ,name:&str) -> Result<T, std::boxed::Box<bincode::ErrorKind>> {
		let f = std::fs::read(format!("{}.token",name).as_str())?;
		bincode::deserialize( &f )
	}
	pub fn sentence_to_str(& self,token_id:Vec<usize>) -> String {
		
		token_id.iter().map(|i|
			format!("{} ",self.token[*i].0) 
		).collect::<String>()
		
	}
	pub fn run (&mut self )  {
		self.matrix.1 = self.bias.0;
		{
			let inp = Simd::from(self.matrix.0);
			self.matrix.1[0].iter_mut().zip(self.weight.0).for_each(|(m,w)|{
				*m = sigmoy!(activation,*m + ( inp * Simd::from(w) ).to_array().iter().sum::<f32>())	
			});
		}
		(1..self.matrix.1.len()).zip(self.weight.1).for_each(|(m,w)| {
			let inp = Simd::from(self.matrix.1[m-1]);
			(0..self.matrix.1[m].len()).for_each(|i| {
				self.matrix.1[m][i] += (inp * Simd::from(self.weight.1[m-1][i])).to_array().iter().sum::<f32>();
				self.matrix.1[m][i] = sigmoy!(activation,self.matrix.1[m][i])
					
			})
			
		});
		
		
		let sum : f32 = self.matrix.1[self.matrix.1.len() - 1 ].iter().sum::<f32>();
		self.matrix.2 = ( Simd::from([sum;OU]) + Simd::from(self.bias.1) ).to_array();
		sigmoy!(output = activation + simd, self.matrix.2);
	
		
	}
	pub fn load_new_token(&mut self) {
		if let Ok(T) = self.load_bincode_token(&self.save_name) {
			self.token = T
		}
	}
	pub fn rand_bias(&mut self){
		let mut rng = rand::thread_rng();
		for i in &mut self.bias.0 {
			for x in i {
				*x = rng.gen();
			}
		}
		for i in &mut self.bias.1 {
			* i = rng.gen();
		}
			
	}
	pub fn rand_weight(&mut self){
		let mut rng = rand::thread_rng();
		for i in &mut self.weight.0 {
			for x in i {
				* x = rng.gen();
			}
		}
		for i in &mut self.weight.1 {
			for x in i {
				for y in x {
					* y = rng.gen();
				}
			}
		}
	
	}
	fn get_random_buf() -> Result<[u8; 32], getrandom::Error> {
		let mut buf = [0u8; 32];
		getrandom::getrandom(&mut buf)?;
		Ok(buf)
	}

	pub fn new (name:&str) -> Neuron {
		
		let mut v : Neuron= Neuron{
			matrix : ([0.0;I],[[0.0;HR];HL],[0.0;OU]),
			bias :([[0.0;HR];HL],[0.0;OU]),
			weight :([[0.0;I];HR],[[[0.0;HR];HR];WE]),
			delta : ([[0.0;HR];HL],[0.0;OU]),
			token : Vec::new(),
			save_name : *Box::new(name.into())
		};
		v.load_new_token();
		
		let name = &v.save_name ;
		if let Ok(T) = v.load_bincode_bias(name) {
			v.bias = T;
		} else {
			v.rand_bias();
			println!("make new bias");
		}
		let name = &v.save_name ;
		
		if let Ok(T) = v.load_bincode_weight(name) {
			v.weight = T;
		} else {
			v.rand_weight();
			println!("make new weight");
		}
		v
	}
}
impl Drop for Neuron {
    fn drop(&mut self) {
		let (bias,weight,token) = self.saving();
		std::thread::scope(|s| {
			let path = &self.save_name;
			let mut file =  self.save_files_bias(path).unwrap();
			s.spawn( move || {
				if let Ok(bias_encoded) = bincode::serialize(&bias) {
					file.write_all(&bias_encoded).unwrap();
				} else {
					panic!()
				}
						
			});
					
			let mut file = self.save_files_weight(path).unwrap();
			s.spawn(move || {
				let weight_encoder : Vec<u8> = bincode::serialize(&weight).unwrap();
				file.write_all(&weight_encoder).unwrap();
			});
					
			let mut file = self.save_files_token(path).unwrap();
		
			let token_encoder : Vec<u8> = bincode::serialize(&token).unwrap();
			file.write_all(&token_encoder).unwrap();
		
		});
				
        println!("Dropping & save");
    }
}

fn main() {
    dioxus_desktop::launch(dioxus_machina );
	//dioxus_web::launch(webapp);
}
fn webapp(cx: Scope)-> Element{
    cx.render(rsx! {
        div {
            "Hello, world!"
        }
    })
}

fn dioxus_machina(cx: Scope) -> Element {
	let machine = use_ref(cx, || { Neuron::new("dist/conscious") });
	let load_local = use_state(cx, || "load local" );
	let  b_download = use_state(cx, || "download unique words" );
	let  b_download_ethics = use_state(cx, || "download ethics" );
	let  b_download_moral = use_state(cx, || "download moral" );
	let  b_save = use_state(cx, || "" );
	let textlog = use_state(cx, || "".to_string() );
	let text= use_state(cx, || "".to_string() );
	
	render!{
		button { onclick: move |_| {
			let mut bind = machine.write();
			let sum1 : f32 = bind.train(Arc::new([([0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0])]) ,1 ).iter().sum();
			let sum2 : f32 = bind.train(Arc::new([([0.0,0.0,0.0,0.0],[1.0,1.0,1.0,1.0])]),1).iter().sum();
			println!("sum1:{}\nsum2:{}\nvalue:good\n",sum1,sum2);
					
		},"run test"}
		button { onclick: move |_| {
			let mut binding = machine.write();
			let path = binding.save_name.clone();
			if let Ok(_) = binding.load_unique_words_from_file(&path) {
				load_local.set("load local(ok)")
			} else {
				load_local.set("load local(error)")
			}
				
		} ,"{load_local}"}
		button { onclick: move |_| {
			b_download.set("(downloading)");
			if let Err( Er ) = machine.write().download_words("https://rifqideveloper.github.io/aichar_train_data/rawwords") {
				b_download.set("download unique words (gagal)")
			} else {
				b_download.set("download unique words (berhasil)")
			}
			
		} ,"{b_download}"}
			
		button { onclick: move |_| {
			b_download_ethics.set("(downloading)");
			match machine.write().download_pages("https://rifqideveloper.github.io/aichar_train_data/ethics.txt") {
				Err( Er ) => {
					b_download_ethics.set("download ethics (gagal)")
				} Ok(token_index) => {
					b_download_ethics.set("download ethics (berhasil)")
				}
			}
			
		} ,"{b_download_ethics}"}
			
		button { onclick: move |_| {
			b_download_moral.set("(downloading)");
			match machine.write().download_pages("https://rifqideveloper.github.io/aichar_train_data/moral.txt") {
				Err( Er ) => {
					b_download_moral.set("download moral (gagal)")
				} Ok(token_index) => {
					b_download_moral.set("download moral (berhasil)")
				}
			}
			
		} ,"{b_download_moral}"}
		button { onclick: move |_| {
			b_save.set("(looding)");
			let binding = machine.read();
			let (bias,weight,token) = binding.saving();
			std::thread::scope(|s| {
				let path : &str = &binding.save_name;
				let mut file = binding.save_files_bias(path).unwrap();
				s.spawn( move || {
					if let Ok(bias_encoded) = bincode::serialize(&bias) {
						file.write_all(&bias_encoded).unwrap();
					} else {
						panic!()
					}
						
				});
					
				let mut file = binding.save_files_weight(path).unwrap();
				s.spawn(move || {
							
					let weight_encoder : Vec<u8> = bincode::serialize(&weight).unwrap();
					file.write_all(&weight_encoder).unwrap();
				});
						
				let mut file = binding.save_files_token(path).unwrap();
				s.spawn(move || {
					let token_encoder : Vec<u8> = bincode::serialize(&token).unwrap();
					file.write_all(&token_encoder).unwrap();
				});
				
				binding.save_unique_words(path).unwrap();
			});
			b_save.set("(berhasil)")
				
		},"save{b_save}"}
		div {
			textarea {
				"{textlog}"
			}
			
		}
		input {
            value: "{text}",
           
            oninput: move |evt| {
				text.set(evt.value.to_lowercase());
			}
        }
		button { onclick: move |_| {
			let mut binding = machine.write();
			
			let (comment,indexs) = binding.str_parse_n_comment( text.get() );
			let repl = binding.dialog_str(indexs.clone());
			textlog.set( format!("{}you:{}\nAI:{}\n",textlog.get(),comment,repl) );
			text.set("".to_string());
		} ,"submit"}
		
			
	}
}
#[cfg(test)]
mod tests {
	use crate::Neuron;
	use crate::Arc;
	use std::cmp::Ordering;
	use std::simd::{Simd,LaneCount,SupportedLaneCount};
	use std::io::Write;
	#[test]
    fn speed_test() {
        let mut machine :Neuron<1,2,2,1,1>= Neuron::new("");
        for _ in 0..1_000_000u32 {
			machine.run();
		}
    }
	#[test]
	fn train_test() {
		let mut machine :Neuron<1,2,2,1,1>= Neuron::new("");
			let sum1 : f32 = machine.train(Arc::new([([0.0],[1.0])]) ,1 ).iter().sum();
			let sum2 : f32 = machine.train(Arc::new([([0.0],[1.0])]),1 ).iter().sum();
			assert_eq!( sum1 < sum2 , true );
	}
	#[test]
    fn parse() {
		//hard error
        let mut machine :Neuron<1,2,2,1,1>= Neuron::new("");
		let test = machine.str_parse("one two . 123".to_string());
			let tst = test.clone();
			assert_eq!( tst[0], 0 );
			assert_eq!( tst[1], 1 );
			assert_eq!( tst[2], 2 );
			assert_eq!( tst[3], 3 );
			assert_eq!( machine.token[test[3]], ("123".to_string(),0.0) ) ;
    }
	#[test]
	fn save_to_file() {
		let machine :Neuron<1,2,2,1,1>= Neuron::new("");
			
			std::thread::scope(|s| {
				let (bias,weight,token) = machine.saving();
				
				let mut file = machine.save_files_bias("testing");
				s.spawn( move || {
					let bias_encoded: Vec<u8> = bincode::serialize(bias).unwrap();
					file.write_all(&bias_encoded).unwrap();
				});
				
				let mut file = machine.save_files_weight("testing");
				s.spawn(move || {
					let weight_encoder : Vec<u8> = bincode::serialize(weight).unwrap();
					file.write_all(&weight_encoder).unwrap();
				});
				
				let mut file = machine.save_files_token("testing");
				s.spawn(move || {
					let token_encoder : Vec<u8> = bincode::serialize(token).unwrap();
					file.write_all(&token_encoder).unwrap();
				});
			});
	}
	#[test]
	fn save_unique() {
		
		let mut machine :Neuron<1,2,2,1,1>= Neuron::new("");
			machine.str_parse("halo my name is ? ".to_string());
			machine.save_sentence("testing",Vec::from(["halo".to_string(),"my".to_string(),"name".to_string(),"is".to_string(),"?".to_string()]) ).unwrap();
			machine.save_unique_words("testing").unwrap();
			assert_eq!( machine.sentence_to_str( machine.load_sentence_from_file("testing") ),"halo my name is ? ".to_string());
	}
	fn download() { 
	
		let mut machine :Neuron<1,2,2,1,1>= Neuron::new("");
			machine.download_words("https://rifqideveloper.github.io/aichar_train_data/rawwords");
		assert_eq!( machine.token[0].0 , "halo".to_string() ) ;
	}
}