#![feature(adt_const_params)]
mod lib;
mod kamus;
fn main() {
	let mut test = lib::Transformer::<{kamus::kamus_line},4,1,0,10>::init("dist/");
	test.training(&["halo".to_string(),"halo".to_string()],&kamus::kata(),10_000);
		//test.print("halo".to_string(),&kamus::kata());
}
