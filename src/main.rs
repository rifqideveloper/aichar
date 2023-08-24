#![feature(portable_simd)]
use std::sync::Arc;
use std::simd::{Simd,LaneCount,SupportedLaneCount};
use std::f32::consts::E;
use std::thread;
use std::sync::mpsc::{Sender, Receiver};
use std::sync::mpsc;
use rand::prelude::{/*thread_rng,*/Rng};
use std::fs::File;
use std::io::Write;
use dioxus::prelude::*;
use serde::{/*Serialize, Deserialize ,*/de};
use std::slice::{Iter,IterMut};
use std::iter::Chain;

macro_rules! sigmoy {
	(activation, $x:expr ) => {
		1.0 / (1.0 + E.powf(-$x))
	};
	(activation + iter, $x:expr ) => {
		$x.iter_mut().for_each(|i| *i = sigmoy!(activation,*i) );
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
struct neuron <const I: usize,const HR: usize,const HL: usize,const WE: usize,const OU: usize>{
    matrix : ([f32;I],[[f32;HR];HL],[f32;OU]),
    weight :Arc<([[f32;I];HR],[[[f32;HR];HR];WE])>,
    bias :Arc<([[f32;HR];HL],[f32;OU])>,
}
impl <const I: usize,const HR: usize,const HL: usize,const WE: usize,const OU: usize> 
	neuron <{I},{HR},{HL},{WE},{OU}>  where LaneCount<I>: SupportedLaneCount,LaneCount<HR>: SupportedLaneCount,LaneCount<HL>: SupportedLaneCount,LaneCount<WE>: SupportedLaneCount,LaneCount<OU>: SupportedLaneCount,
{
    fn input_neuron_caculate(&mut self) {
        for i in 0..HL {
            self.matrix.1[0][i] = (Simd::from(self.matrix.0) * Simd::from(self.weight.0[1]) ).to_array().iter().sum();    
        }
        self.matrix.1[0] = (Simd::from(self.matrix.1[0]) + Simd::from(self.bias.0[0]) ).to_array();
    }
    fn hidden_neuron_caculate_sigmoy(&mut self) {
        for i in 1..HL {
            let input = i - 1;
            for x in 0..HR {
                self.matrix.1[i][x] = (Simd::from(self.matrix.1[input]) * Simd::from(self.weight.1[input][i-1]) ).to_array().iter().sum();
            }
            self.matrix.1[i] = (Simd::from(self.matrix.1[i]) + Simd::from(self.bias.0[i]) ).to_array();
            sigmoy!(activation + iter, self.matrix.1[i] );
        }
        
    }
    fn output_neuron_caculate(&mut self) {
        self.matrix.2 = (Simd::from(self.matrix.2) + Simd::from(self.bias.1) ).to_array()
    }
    pub fn run(&mut self) -> [f32;OU] {
        self.input_neuron_caculate();
        sigmoy!(activation + iter, self.matrix.1[0] );
        self.hidden_neuron_caculate_sigmoy();
        self.output_neuron_caculate();
        sigmoy!(output = activation + simd , self.matrix.2);
        self.matrix.2
    }
}

const VecSBlen : usize = 500;
#[derive(Clone,PartialEq,Debug)]
struct VecSB {
	Main:[(Box<str>,f32,f32,f32);VecSBlen],
	Extra:Vec<(Box<str>,f32,f32,f32)>
}
impl VecSB {
	pub fn nonegativ(&self,x:f32) -> f32 {
		x * ((x > 0.0) as i32 - (x < 0.0) as i32) as f32
	}
	pub fn iter(& self) -> Chain<Iter<'_, (Box<str>, f32, f32, f32)>, Iter<'_, (Box<str>, f32, f32, f32)>> {
		let v : Iter<'_, (Box<str>, f32, f32, f32)> = self.Extra.iter();
		let vv : Iter<'_, (Box<str>, f32, f32, f32)> = self.Main.iter();
		vv.chain(v)
	}
	pub fn iter_mut(& mut self) -> Chain<IterMut<'_, (Box<str>, f32, f32, f32)>, IterMut<'_, (Box<str>, f32, f32, f32)>> {
		
		let v : IterMut<'_, (Box<str>, f32, f32, f32)> = self.Extra.iter_mut();
		let vv : IterMut<'_, (Box<str>, f32, f32, f32)> = self.Main.iter_mut();
		vv.chain(v)
	}
	pub fn get<'a>(&'a self,index:usize)-> &'a (Box<str>,f32,f32,f32) {
	    if self.Main.len() <= index {
	        return &self.Extra[index-self.Main.len()]
	    } 
	    &self.Main[index]
	}
	pub fn get_mut<'a>(&'a mut self,index:usize)-> &'a mut (Box<str>,f32,f32,f32) {
	    if self.Main.len() <= index {
	        return &mut self.Extra[index-self.Main.len()]
	    } 
	    &mut self.Main[index]
	}
	pub fn get_coordinat<'a>(&'a self,token:(f32,f32,f32) ) -> &'a (Box<str>,f32,f32,f32) {
		let mut pcoordinat : &(Box<str>,f32,f32,f32)= &self.Main[0];
		let mut komp : f32 = self.nonegativ([self.Main[0].1 - token.0,self.Main[0].2 - token.1,self.Main[0].2 - token.1].iter().sum());
		for i in 1..self.Main.len() {
		    let komp2 : f32 = self.nonegativ([self.Main[i].1 - token.0,self.Main[i].2 - token.1,self.Main[i].2 - token.1].iter().sum());	
		
		    if komp > komp2 {
				komp = komp2;
				pcoordinat = &self.Main[i];
			}
		}
		for i in &self.Extra {
			let komp2 : f32 = self.nonegativ([i.1 - token.0,i.2 - token.1,i.2 - token.1].iter().sum());
		
		    if komp > komp2 {
				komp = komp2;
				pcoordinat = &i;
			}
		}
		pcoordinat
	}
	pub fn rposition(&self,token:&str) -> Option<usize> {
		for i in 0..self.Main.len() + self.Extra.len() {
			if VecSBlen > i {
				if *self.Main[i].0 == *token {
					//println!("{}=={}",self.Main[i].0,token);
					return Some(i);
				}
			} else {
				let index = i - VecSBlen ;
				if *self.Extra[index].0 == *token {
					//println!("{}=={}:{}",self.Extra[index].0,token,i);
					return Some(i);
				}
			}
			
		}
		None
	}
	pub fn pop(&mut self) {
		self.Extra.pop();
	}
	pub fn push(&mut self,v:(Box<str>,f32,f32,f32)){
		self.Extra.push(v)
	}
	pub fn len(&self) -> usize {
		self.Main.len() + self.Extra.len()
	}
	pub fn collect(&self) -> Vec<(Box<str>,f32,f32,f32)> {
        let mut v = Vec::with_capacity(self.len());
        for i in &self.Main {
            v.push(i.clone())
        }
        for i in &self.Extra {
            v.push(i.clone())
        }
        v
    }
	pub fn from(v:Vec<(Box<str>,f32,f32,f32)>) -> VecSB {
		let mut v = v.iter();
		
		let Main : [(Box<str>,f32,f32,f32);VecSBlen] = (0..VecSBlen).map(|_| v.next().unwrap().clone() ).collect::<Vec<_>>().try_into().unwrap();
		let Extra : Vec<(Box<str>,f32,f32,f32) >= v.map(|v| v.clone() ).collect();
		
		VecSB {
			Main:Main,
			Extra:Extra
		}
		
	}
}
const MAIN_I : usize = 4;
const MAIN_HR : usize = 2;//2;
const MAIN_HL : usize = 2;//2;
const MAIN_WE : usize = 1;//1;
const MAIN_OU : usize = 4;

#[derive(Debug)]
struct neuron_builder{
	weight :Arc<([[f32;MAIN_I];MAIN_HR],[[[f32;MAIN_HR];MAIN_HR];MAIN_WE])>,
    bias :Arc<([[f32;MAIN_HR];MAIN_HL],[f32;MAIN_OU])>,
	delta : ([[f32;MAIN_HR];MAIN_HL],[f32;MAIN_OU]),
	token : Option<VecSB>,
	save_name : Box<str>,
	neuron:Vec<(Sender<Option<[f32;MAIN_I]>>,Receiver<[f32;MAIN_OU]>,thread::JoinHandle<()>)>
}
impl neuron_builder {
	pub fn nonegativ(&self,x:f32) -> f32 {
		x * ((x > 0.0) as i32 - (x < 0.0) as i32) as f32
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
	pub fn koordinat_to_str(& self,buf:&mut String,koordinat:[f32;MAIN_OU])  {
		let mut komp : f32 = sigmoy!(activation,[self.token.as_ref().unwrap().get(0).1 - koordinat[0],self.token.as_ref().unwrap().get(0).2- koordinat[1],self.token.as_ref().unwrap().get(0).3- koordinat[2]].iter().sum::<f32>() );
		let mut v : &str = &self.token.as_ref().unwrap().get(0).0;
		
		for i in 1..self.token.as_ref().unwrap().len() {
			let a : f32 = sigmoy!(activation, [self.token.as_ref().unwrap().get(i).1 - koordinat[0],self.token.as_ref().unwrap().get(i).2- koordinat[1],self.token.as_ref().unwrap().get(i).3- koordinat[2]].iter().sum::<f32>() );
			if komp > a {
				komp = a;
				v = &self.token.as_ref().unwrap().get(i).0;
			}
		}
		
		buf.push_str(v);
		buf.push(' ');
	}
	pub fn dialog_str(&mut self,text:Arc<[usize]>) -> String{
		let mut dialog = "".to_string();
		let mut mode : f32 = 0.0;
		for index in 0..text.len() {
			if self.neuron.len() == index {
				self.addneuron();
			}
			let test = self.token.as_ref().unwrap().get(text[index]);
			
			self.neuron[index].0.send(Some([mode,test.1,test.2,test.3])).unwrap();
		}
		for i in 0..text.len() {
			let test = self.neuron[i].1.recv().unwrap();
			self.koordinat_to_str(&mut dialog,test);
		}
		dialog
	}
	pub fn str_parse_n_comment(&mut self,text:&str) -> (String,Arc<[usize]>) {
		let mut comment = String::new();
		
		let v : Arc<[usize]> = self.strsplit(text).iter().map(|new_token| {
			if let Some(i) = self.token.as_ref().unwrap().rposition(new_token) {
				comment.push_str(&format!("{} ",self.token.as_ref().unwrap().get(i).0));
				i
			} else {
				let mut rng = rand::thread_rng();
				let v = self.token.as_ref().unwrap().len();
				comment.push_str(&format!("{} ",new_token));
				self.token.as_mut().unwrap().push((new_token.as_str().into(),rng.gen(),rng.gen(),rng.gen()));
				v
			}
		}).collect();
		
		(comment,v)
	}
	fn addneuron(&mut self) {
		let bias = self.bias.clone();
		let weight = self.weight.clone();
		let (input,input_input) = mpsc::channel();
		let (output,output_output) = mpsc::channel();
		self.neuron.push((
			input,
			output_output,
			thread::spawn(move || {
				
				let mut  neuron : neuron<MAIN_I,MAIN_HR,MAIN_HL,MAIN_WE,MAIN_OU> = neuron {
					matrix : ([0.0;MAIN_I],[[0.0;MAIN_HR];MAIN_HL],[0.0;MAIN_OU]),
					bias:bias,
					weight:weight
				};
				loop{
					if let Some(_input) = input_input.recv().unwrap() {
						neuron.matrix.0 = _input;
						neuron.run();
						output.send(neuron.matrix.2).unwrap();
					} else {
						break
					}
				}
				
				
				
			})
		))
	}
	fn send_input(&mut self,index:usize,input:[f32;MAIN_I]) {
		if let Some(T) = self.neuron.get_mut(index) {
			&mut T.0
		} else {
			self.addneuron();
			&mut self.neuron[index].0
		}.send(Some(input)).unwrap();
		
	}
	pub fn saving<'a>( &'a self ) -> (
		Arc<([[f32;MAIN_HR];MAIN_HL],[f32;MAIN_OU])>,
		Arc<([[f32;MAIN_I];MAIN_HR],[[[f32;MAIN_HR];MAIN_HR];MAIN_WE])>,
		//&'a Vec<(Box<str>,f32,f32,f32)>,
		&'a VecSB
	) { 
		(self.bias.clone(),self.weight.clone(), self.token.as_ref().unwrap() )
	}
	pub fn save_files_bias(& self,file_name:&str) -> Result< File, std::boxed::Box<std::io::Error> > {
		Ok( File::create(format!("{}.bias.{}.{}.{}",file_name,MAIN_HR,MAIN_HL,MAIN_OU).as_str())? )
	}
	pub fn save_files_weight(& self,file_name:&str) -> Result< File, std::boxed::Box<std::io::Error> > {
		Ok( File::create(format!("{}.weight.{}.{}.{}",file_name,MAIN_I,MAIN_HR,MAIN_WE).as_str())? )
	}
	pub fn save_files_token(& self,file_name:&str)-> Result< File, std::boxed::Box<std::io::Error> > {
		Ok( File::create(format!("{}.token",file_name).as_str())? )
		//file_token.write_all(&token_encoder).unwrap();
	}
	/*
	pub fn get_bias_bincode(name:&str) -> Result< ([[f32;MAIN_HR];MAIN_HL],[f32;MAIN_OU]), std::boxed::Box<std::io::Error> > {
		bincode::deserialize(&std::fs::read(&format!("{}.bias.{}.{}.{}",name,MAIN_HR,MAIN_HL,MAIN_OU))?)
	}
	pub fn get_weight_bincode(name:&str) -> Result< ([[0.0f32;MAIN_I];MAIN_HR],[[[0.0f32;MAIN_HR];MAIN_HR];MAIN_WE]), std::boxed::Box<std::io::Error> > {
		bincode::deserialize(&std::fs::read(&format!("{}.weight.{}.{}.{}",name,MAIN_I,MAIN_HR,MAIN_WE))?)
	}
	*/
	pub fn new(name:&str) -> neuron_builder {
		let mut nb = neuron_builder{
			weight : if let Ok(T) =bincode::deserialize(&std::fs::read(&format!("{}.weight.{}.{}.{}",name,MAIN_I,MAIN_HR,MAIN_WE)).unwrap()) {
					Arc::new(T)
				} else {
					let mut rng = rand::thread_rng();
					let mut T = ([[0.0f32;MAIN_I];MAIN_HR],[[[0.0f32;MAIN_HR];MAIN_HR];MAIN_WE]);
					for i in &mut T.0 {
						for x in i {
							* x = rng.gen();
						}
					}
					for i in &mut T.1 {
						for x in i {
							for y in x {
								* y = rng.gen();
							}
						}
					}
					Arc::new(T)
				},
			bias : if let Ok(T) = bincode::deserialize(&std::fs::read(&format!("{}.bias.{}.{}.{}",name,MAIN_HR,MAIN_HL,MAIN_OU)).unwrap()){
					Arc::new(T)
				} else{
					let mut rng = rand::thread_rng();
					let mut T = ([[0.0f32;MAIN_HR];MAIN_HL],[0.0f32;MAIN_OU]);
					
					for i in &mut T.0 {
						for x in i {
							*x = rng.gen();
						}
					}
					for i in &mut T.1 {
						* i = rng.gen();
					}
					Arc::new(T)
				},
			delta : ([[0.0;MAIN_HR];MAIN_HL],[0.0;MAIN_OU]),
			token : if let Ok(T) = bincode::deserialize(&std::fs::read(&format!("{}.token",name)).unwrap()) {
					Some(VecSB::from(T))
				} else{
					None
				},
			save_name : name.into(),
			neuron:Vec::with_capacity(5)
		};
		nb.addneuron();
		/*//tes
		nb.send_input(0,[0.0;MAIN_I]);
		nb.neuron[0].1.recv().unwrap();
		*/
		//print!("{:?}",nb);
		nb
	}
}
impl Drop for neuron_builder {
	fn drop(&mut self) {
		let (bias,weight,token) = self.saving();
		std::thread::scope(|s| {
			let path = &self.save_name;
			let mut file =  self.save_files_bias(path).unwrap();
			s.spawn( move || {
				if let Ok(bias_encoded) = bincode::serialize(&*bias) {
					file.write_all(&bias_encoded).unwrap();
				} else {
					panic!()
				}
						
			});
					
			let mut file = self.save_files_weight(path).unwrap();
			s.spawn(move || {
				let weight_encoder : Vec<u8> = bincode::serialize(&*weight).unwrap();
				file.write_all(&weight_encoder).unwrap();
			});
					
			let mut file = self.save_files_token(path).unwrap();
		
			let token_encoder : Vec<u8> = bincode::serialize(&token.collect()).unwrap();
			file.write_all(&token_encoder).unwrap();
		
		});
		for _ in 0..self.neuron.len() {
			let mut j = self.neuron.pop().unwrap();
				j.0.send(None).unwrap();
				j.2.join().unwrap();
			
		}
	}
}
fn main () {
    dioxus_desktop::launch( dioxus_machina );
}

fn dioxus_machina(cx: Scope) -> Element {
	let machine = use_ref(cx, || unsafe { 
		neuron_builder::new("dist/conscious" ) 
	});
	let textlog = use_state(cx, || "".to_string() );
	let text= use_state(cx, || "".to_string() );
	
	render!{
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
			let binding = &mut machine.write();
			let (comment,indexs) = binding.str_parse_n_comment( text.get() );
			let repl = binding.dialog_str(indexs.clone());
			textlog.set( format!("{}you:{}\nAI:{}\n",textlog.get(),comment,repl) );
			text.set("".to_string());
		},"submit"}
	}
}