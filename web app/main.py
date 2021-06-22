import streamlit as st 

from sklearn import datasets






 
 
 
def main():
    st.title("blood pressur")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">blood pressur estimation ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write("select model")
    modelname=st.selectbox("select model",("DNN","Hybrid"))
    
    dataset= st.file_uploader("upload dataset",type=["csv","txt","json"])
    
   
    result=""
    if st.button("Predict"):
        result=
    st.success('The output is {}'.format(result))
   

if __name__=='__main__':
    main()

 

