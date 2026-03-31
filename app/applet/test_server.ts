import express from 'express';
import multer from 'multer';
import { Jimp } from 'jimp';

const app = express();
app.get('/test', (req, res) => res.send('ok'));
app.listen(3001, () => console.log('listening'));
