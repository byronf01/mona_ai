/*
Localhost server used to store conversational prompt data
*/

const express = require('express')
const app = express();
const path = '../data_mona/data.json'
console.log(data)

app.listen(5001, () => {console.log("Server started on port 5001")})

app.get("/data", async (req, res) => {
    // Return all conversational mona data

    
    
    fetch(path).then( resp => { return resp.json() } )

})

