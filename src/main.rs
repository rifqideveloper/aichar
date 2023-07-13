#![allow(non_snake_case)]
#![feature(portable_simd)]

use dioxus::prelude::*;
use std::default::Default;
use std::sync::Arc;
use std::f32::consts::E;
use std::simd::{Simd,LaneCount,SupportedLaneCount};
use rand::prelude::{thread_rng,Rng};

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

#[derive(Debug,PartialEq,Clone)]
struct Neuron <const I: usize,const HR: usize,const HL: usize,const WE: usize,const OU: usize>  {
	matrix : ([f32;I],[[f32;HR];HL],[f32;OU]),
	bias :([[f32;HR];HL],[f32;OU]),
	weight :([[f32;I];HR],[[[f32;HR];HR];WE]),
	delta : ([[f32;HR];HL],[f32;OU]),
	token : Vec<(String,f32)>
}
impl<const I: usize,const HR: usize,const HL: usize,const WE: usize,const OU: usize>  
Neuron <{I},{HR},{HL},{WE},{OU}>  where LaneCount<I>: SupportedLaneCount,LaneCount<HR>: SupportedLaneCount,LaneCount<HL>: SupportedLaneCount,LaneCount<WE>: SupportedLaneCount,LaneCount<OU>: SupportedLaneCount,
{
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
				//self.update_weight(0.1);
				total_error.push( LEARNING_RATE );
			}
		}
		total_error
	}
	pub fn learning(&mut self ,expected:&[f32;OU]) -> f32 {
		let mut sum_error = 0.0;
		self.delta.1 = (Simd::from(self.matrix.2) - Simd::from(*expected) ).to_array() ;
		self.delta.1.iter_mut()
			.for_each(|delta| {*delta = sigmoy!(derivative, *delta ) ;sum_error += *delta;} );
			
		self.delta.0 = [[0.0;HR];HL];
		
		for j in 0..self.matrix.1[0].len() {
			for weight in self.weight.1[WE-1][j] {
				for delta in 0..self.delta.1.len() {
					self.delta.0[HL-1][j] -= weight * self.delta.1[delta]
				}
			} 
		}
		let mut derivative : [f32;HR] = self.matrix.1[HL-1].iter()
			.map(|x| sigmoy!(derivative,x) )
			.collect::<Vec<_>>().try_into().unwrap();
		self.delta.0[HL-1] = ( Simd::from(derivative) * Simd::from(self.delta.0[HL-1]) ).to_array();
		
		for i in (1..HL-1).rev() {
			for j in 0..self.matrix.1[0].len() {
				for weight in self.weight.1[i-1][j] {
					for delta in 0..self.delta.0[i+1].len() {
						self.delta.0[i][j] -= weight * self.delta.0[i+1][delta]
					}
				} 
			}
			let mut derivative : [f32;HR] = self.matrix.1[i].iter()
				.map(|x| sigmoy!(derivative,x) )
				.collect::<Vec<_>>().try_into().unwrap();
			self.delta.0[i] = ( Simd::from(derivative) * Simd::from(self.delta.0[i]) ).to_array()
		}
		for j in 0..self.matrix.1[0].len() {
			for weight in self.weight.0[j] {
				for delta in 0..self.delta.0[1].len() {
					self.delta.0[0][j] -= weight * self.delta.0[1][delta]
				}
			} 
		}
		let mut derivative : [f32;HR] = self.matrix.1[0].iter()
			.map(|x| sigmoy!(derivative,x) )
			.collect::<Vec<_>>().try_into().unwrap();
		self.delta.0[0] = ( Simd::from(derivative) * Simd::from(self.delta.0[0]) ).to_array();
		
		sum_error 
	}
	pub fn update_weight(&mut self ,LEARNING_RATE : f32) {
		for i in 0..self.delta.0[0].len() {
			for x in 0..self.weight.0[i].len() {
				self.weight.0[i][x] += LEARNING_RATE * self.delta.0[0][i] * self.matrix.0[x];
			}
		}
		let offset = self.matrix.1.len() - 1;
		for offset in 1..offset {
			for i in 0..self.matrix.1[offset].len() {	
				for x in 0..self.weight.1[offset-1][i].len() {
					self.weight.1[offset-1][i][x] += LEARNING_RATE * self.delta.0[offset][i] * self.matrix.1[offset-1][i]
				}
			}
		}
		for i in 0..self.matrix.1[offset].len() {	
			for x in 0..self.weight.1[offset-1][i].len() {
				self.weight.1[offset-1][i][x] += LEARNING_RATE * self.delta.0[offset][i] * self.matrix.1[offset-1][i]
			}
		}
		
	}
	pub fn str_parse(&mut self,text:String) -> Vec<usize> {
		text.split_whitespace().map(|i| {
			if let Some(i) = self.token.iter().rposition(|x| x.0 == i ) {
				i
			} else {
				let v = self.token.len();
				self.token.push((i.to_string(),0.0));
				v
			}
		}).collect()
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
	
	pub fn new  (name:&str) -> Neuron <I, HR, HL,WE, OU> {
		let mut v : Neuron<I, HR, HL,WE, OU>= Neuron{
			matrix : ([0.0;I],[[0.0;HR];HL],[0.0;OU]),
			bias :([[0.0;HR];HL],[0.0;OU]),
			weight :([[0.0;I];HR],[[[0.0;HR];HR];WE]),
			delta : ([[0.0;HR];HL],[0.0;OU]),
			token : Vec::new()			
		};
		if !name.is_empty() {
			if false {
				return v
			}
		}
		let mut rng = rand::thread_rng();
		for i in &mut v.bias.0 {
			for x in i {
				*x = rng.gen();
			}
		}
		for i in &mut v.bias.1 {
			* i = rng.gen();
		}
		for i in &mut v.weight.0 {
			for x in i {
				* x = rng.gen();
			}
		}
		for i in &mut v.weight.1 {
			for x in i {
				for y in x {
					* y = rng.gen();
				}
			}
		}
		v
	}
}
fn main() {
    dioxus_desktop::launch(dioxus_machina);
}


fn dioxus_machina(cx: Scope) -> Element {
	let machine = use_ref(cx, || Neuron::<1,2,2,1,1>::new("") );
	render!{
		button { onclick: move |_| {
			let mut machine :Neuron<1,2,2,1,1>= Neuron::new("");
				let sum1 : f32 = machine.train(Arc::new([([0.0],[1.0])]) ,1 ).iter().sum();
				let sum2 : f32 = machine.train(Arc::new([([0.0],[1.0])]),1).iter().sum();
				println!("sum1:{}\nsum2:{}",sum1,sum2);
		},"run"}
        div {
            "Hello, world!"
        }
	}
}

#[cfg(test)]
mod tests {
	use crate::Neuron;
	use crate::Arc;
	use std::cmp::Ordering;
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
			let sum2 : f32 = machine.train(Arc::new([([0.0],[1.0])]),1).iter().sum();
			assert_eq!( sum1 < sum2 , true );
	}
	#[test]
    fn parse() {
        let mut machine :Neuron<1,2,2,1,1>= Neuron::new("");
		let test = machine.str_parse("one two . 123".to_string());
			assert_eq!( test, Vec::from([0,1,2,3]) );
			assert_eq!( machine.token[test[3]], ("123".to_string(),0.0) ) ;
    }
}
