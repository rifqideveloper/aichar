#![feature(const_trait_impl)]
use std::thread;
use std::sync::mpsc;
<<<<<<< HEAD
use std::thread::JoinHandle;
=======
>>>>>>> 68b8235e9f8befc241e2465061c0ab2104bc3b34
use std::sync::mpsc::Sender;
use std::time::{Duration, SystemTime};
use std::f32::consts::E;
//static mut BIASE  : Vec<f32> = Vec::new();
static mut WEIGHT : Vec<f32> = Vec::new();
static mut INPUT : [Vec<f32>;3] = [Vec::new(),Vec::new(),Vec::new()];
static mut tr : Vec<JoinHandle<()>> = Vec::new();
fn sigmoy(x:f32) -> f32 {
	1.0 / (1.0 + E.powf(-x))
}
fn matrix(input:&[f32],hidden:&[(f32,f32/*biase*/,&[(usize,usize)],bool)],output:(usize,&[f32])) {
	unsafe {
		INPUT[0] = Vec::from(input);
		INPUT[1].clear();
		for i in 0..hidden.len() {
			if WEIGHT.len() == i {
				WEIGHT.push(hidden[i].0);
			} else {
				WEIGHT[i] = hidden[i].0;
			}
			INPUT[1].push(hidden[i].1);
			
			let v = Vec::from(hidden[i].2);
			if hidden[i].3 {
				tr.push(thread::spawn(move || {
					for x in v.iter() {
						while !tr[x.1].is_finished() {}
						INPUT[1][i] += INPUT[x.0][x.1] * WEIGHT[i];
					}
					INPUT[1][i] = sigmoy(INPUT[1][i]);
				}));
			} else {
				tr.push(thread::spawn(move || {
					for x in v.iter() {
						INPUT[1][i] += INPUT[x.0][x.1] * WEIGHT[i];
					}
					INPUT[1][i] = sigmoy(INPUT[1][i]);
				}));
			}
			
		} 
		
		INPUT[2] = Vec::from(output.1);
		
		for thread_ in output.0..tr.len() {
			tr.push(thread::spawn(move || {
				while !tr[thread_].is_finished() {}
				for i in 0..INPUT[2].len() {
					INPUT[2][i] += INPUT[1][thread_] ;
				}
			}));
		}
		for i in 0..output.1.len() {
			let indx = output.1.len() - 1 - i;
			tr.pop().unwrap().join().unwrap() ;
			INPUT[2][indx] = sigmoy(INPUT[2][indx]);
		}
		tr.clear();
	}
}
fn main() {
	unsafe {
		let now = SystemTime::now();
		let imp : &[f32] = &[1.];
		let neuron : &[(f32,f32,&[(usize,usize)],bool)] = &[
					(1.,1.,&[(0,0)] ,false),(1.,1.,&[(0,0)] ,false),(1.,1.,&[(0,0)],false)
					,(1.,1.,&[(1,0),(1,1),(1,2)] ,true)
		];
		let out : (usize,&[f32]) = (3,&[0.,0.]);
		for _ in 0..1000 {
			matrix(
				imp,
				neuron,
				out
			);
			println!("{:?}",INPUT);
			
		}
		match now.elapsed() {
			Ok(elapsed) => {
					
				println!(" time :{}", elapsed.as_millis());
			}
			Err(e) => {
				// an error occurred!
				println!("Error: {e:?}");
			}
		}
	}
    
}
